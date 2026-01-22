"""
Image Interconversion GUI Database Module
=========================================

Provides database functionalities for the Image Interconversion GUI application, including:
- Loading configurations
- Establishing database connections
- Executing queries
- Managing image data with compression, encryption, storage, and retrieval

Utilizes asynchronous programming with aiosqlite for SQLite database interaction and incorporates encryption and compression for secure data handling.

Author: Lloyd Handyside
Contact: ace1928@gmail.com
Version: 1.0.0
Creation Date: 2024-04-06
Last Modified: 2024-04-09
Last Reviewed: 2024-04-09

Functionalities:
- Load database configurations
- Establish database connections
- Execute database queries
- Manage image data with security measures

"""

import asyncio
import aiosqlite
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
from typing import Optional, List, Tuple, AsyncContextManager, NoReturn
import backoff
from contextlib import asynccontextmanager
import io
from PIL import Image  # Import PIL.Image for image handling

# Import ConfigManager, LoggingManager, EncryptionManager from core_services
from core_services import ConfigManager, LoggingManager, EncryptionManager
from image_processing import (
    validate_image,
    compress,
    decompress,
    ensure_image_format,
    encrypt,
    decrypt,
)

__all__ = [
    "load_database_configurations",
    "get_db_connection",
    "execute_db_query",
    "validate_pagination_params",
    "get_images_metadata",
    "insert_compressed_image",
    "retrieve_compressed_image",
    "migrate_schema",
    "init_db",
    "DatabaseConfig",
    "run_tests",
]


class DatabaseConfig:
    """
    Holds the configuration for the database connection and encryption key file path.

    Attributes:
        DB_PATH (str): Path to the SQLite database file.
        KEY_FILE (str): Path to the encryption key file.
    """

    DB_PATH: str = "image_db.sqlite"
    KEY_FILE: str = "encryption.key"


async def load_database_configurations():
    """
    Asynchronously loads database configurations from a config.ini file.

    Updates DatabaseConfig class attributes with values from the configuration file.

    Raises:
        FileNotFoundError: If the config.ini file does not exist.
        KeyError: If essential configuration keys are missing.
        Exception: For any other unexpected errors.

    Example:
        await load_database_configurations()
    """
    LoggingManager.debug("Starting to load database configurations.")
    try:
        config_manager = ConfigManager()
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        await config_manager.load_config(config_path, "Database", file_type="ini")
        DatabaseConfig.DB_PATH = config_manager.get(
            "Database", "db_path", "db_path", fallback="image_db.sqlite"
        )
        DatabaseConfig.KEY_FILE = config_manager.get(
            "Database", "key_file_path", "key_file_path", fallback="encryption.key"
        )
        LoggingManager.info("Database configurations loaded successfully.")
    except FileNotFoundError as e:
        LoggingManager.error(f"Configuration file not found: {e}")
        raise
    except KeyError as e:
        LoggingManager.error(f"Missing essential configuration key: {e}")
        raise
    except Exception as e:
        LoggingManager.error(f"Failed to load database configurations: {e}")
        raise


@asynccontextmanager
async def get_db_connection() -> AsyncContextManager[aiosqlite.Connection]:
    """
    Creates an asynchronous context manager for managing database connections.

    Yields:
        aiosqlite.Connection: An open connection to the database.

    Raises:
        Exception: If connecting to the database fails.
    """
    LoggingManager.debug("Attempting to connect to the database.")
    try:
        database_connection = await aiosqlite.connect(DatabaseConfig.DB_PATH)
        LoggingManager.info("Database connection established successfully.")
        yield database_connection
    except Exception as e:
        LoggingManager.error(f"Failed to connect to the database: {e}")
        raise
    finally:
        await database_connection.close()
        LoggingManager.debug("Database connection closed.")


encryption_key = EncryptionManager.get_valid_encryption_key()
cipher_suite = Fernet(encryption_key)


@backoff.on_exception(
    backoff.expo,
    aiosqlite.OperationalError,
    max_time=60,
    on_backoff=lambda details: LoggingManager.warning(
        f"Retrying due to: {details['exception']}"
    ),
)
# Implements exponential backoff retry for operational errors in database queries.
# Retries for up to 60 seconds before giving up, logging warnings on each retry.
async def execute_db_query(query: str, parameters: tuple = ()) -> aiosqlite.Cursor:
    """
    Executes a database query asynchronously.

    This function executes a given SQL query with the provided parameters. It uses an exponential backoff strategy for retries in case of operational errors.

    Parameters:
        query (str): The SQL query to execute.
        parameters (tuple): A tuple of parameters for the SQL query.

    Returns:
        aiosqlite.Cursor: The cursor resulting from the query execution.

    Raises:
        aiosqlite.OperationalError: If an operational error occurs during query execution.
        Exception: For any other unexpected errors.
    """
    LoggingManager.debug(
        f"Executing database query: {query} with parameters: {parameters}"
    )
    try:
        async with get_db_connection() as db:
            cursor = await db.execute(query, parameters)
            await db.commit()
            LoggingManager.info(f"Query executed successfully: {query}")
            return cursor
    except aiosqlite.OperationalError as e:
        LoggingManager.error(f"Database operational error: {e}, Query: {query}")
        raise
    except Exception as e:
        LoggingManager.error(
            f"Unexpected error executing database query: {e}, Query: {query}"
        )
        raise


def validate_pagination_params(offset: int, limit: int) -> NoReturn:
    """
    Validates the pagination parameters for database queries.

    Ensures that the offset is non-negative and the limit is positive, raising appropriate exceptions if not.

    Parameters:
        offset (int): The offset from where to start fetching the records.
        limit (int): The maximum number of records to fetch.

    Raises:
        TypeError: If either offset or limit is not an integer.
        ValueError: If offset is negative or limit is not positive.
    """
    if not isinstance(offset, int) or not isinstance(limit, int):
        raise TypeError("Offset and limit must be integers.")
    if offset < 0 or limit <= 0:
        raise ValueError("Offset must be non-negative and limit must be positive.")


async def get_images_metadata(
    offset: int = 0, limit: int = 10
) -> List[Tuple[str, str]]:
    """
    Retrieves metadata for images stored in the database within specified pagination parameters.

    This function fetches the hash and format for images stored in the database, limited by the provided offset and limit parameters for pagination purposes.

    Parameters:
        offset (int): The offset from where to start fetching the records. Defaults to 0.
        limit (int): The maximum number of records to fetch. Defaults to 10.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the hash and format of the images.

    Raises:
        Exception: If retrieving the metadata fails.
    """
    validate_pagination_params(offset, limit)
    query = "SELECT hash, format FROM images LIMIT ? OFFSET ?"
    async with get_db_connection() as db:
        cursor = await db.execute(query, (limit, offset))
        result = await cursor.fetchall()
    return result


async def insert_compressed_image(hash: str, format: str, data: bytes) -> None:
    """
    Inserts a compressed and encrypted image into the database.

    This function ensures the image format, validates the image, compresses, and encrypts the data before inserting it into the database.

    Parameters:
        hash (str): The unique hash identifier for the image.
        format (str): The format of the image (e.g., 'png', 'jpg').
        data (bytes): The raw image data to be compressed and encrypted.

    Raises:
        Exception: If the image fails to be inserted into the database.
    """
    LoggingManager.debug(f"Inserting compressed image with hash: {hash}")
    try:
        image = ensure_image_format(data)
        if not validate_image(image):
            LoggingManager.error(f"Data validation failed for image {hash}.")
            return
        compressed_data = await compress(data, format)
        encrypted_data = await encrypt(compressed_data)
        query = "INSERT OR REPLACE INTO images (hash, format, compressed_data) VALUES (?, ?, ?)"
        await execute_db_query(query, (hash, format, encrypted_data))
        LoggingManager.info(f"Image {hash} inserted successfully.")
    except Exception as e:
        LoggingManager.error(f"Failed to insert image {hash}: {e}")
        raise


async def retrieve_compressed_image(hash: str) -> Optional[tuple]:
    """
    Retrieves a compressed and encrypted image from the database.

    This function fetches the compressed and encrypted image data from the database using the provided hash, decrypts, and decompresses it before returning.

    Parameters:
        hash (str): The unique hash identifier for the image to retrieve.

    Returns:
        Optional[tuple]: A tuple containing the decompressed image data and its format, or None if the image is not found.

    Raises:
        Exception: If retrieving the image fails.
    """
    LoggingManager.debug(f"Retrieving compressed image with hash: {hash}")
    query = "SELECT compressed_data, format FROM images WHERE hash = ?"
    try:
        async with get_db_connection() as db:
            cursor = await db.execute(query, (hash,))
            result = await cursor.fetchone()
            if result:
                compressed_data, format = result
                decrypted_data = await decrypt(compressed_data)
                data = await decompress(decrypted_data)
                LoggingManager.info(f"Image {hash} retrieved successfully.")
                return data, format
            LoggingManager.warning(f"No image found with hash: {hash}")
            return None
    except Exception as e:
        LoggingManager.error(f"Failed to retrieve image {hash}: {e}")
        raise


async def migrate_schema():
    """
    Migrates the database schema to the latest version.

    This function checks the current database schema against the expected schema and applies any necessary migrations to bring it up to date.

    Raises:
        Exception: If schema migration fails.
    """
    expected_schema = {
        "images": [
            ("hash", "TEXT PRIMARY KEY"),
            ("format", "TEXT NOT NULL"),
            ("compressed_data", "BLOB NOT NULL"),
        ]
    }

    async with get_db_connection() as db:
        for table, expected_columns in expected_schema.items():
            cursor = await db.execute(f"PRAGMA table_info({table})")
            current_columns = {info[1]: info[2] for info in await cursor.fetchall()}
            missing_columns = [
                col for col in expected_columns if col[0] not in current_columns
            ]

            for col_name, col_type in missing_columns:
                alter_query = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                await db.execute(alter_query)
                LoggingManager.info(
                    f"Added missing column '{col_name}' to '{table}' table."
                )

        await db.commit()
        LoggingManager.info("Database schema migration completed successfully.")


async def init_db():
    """
    Initializes the database and migrates the schema to the latest version.

    This function ensures that the database schema is up to date by calling the migrate_schema function. It is intended to be run at application startup to prepare the database for use.

    Raises:
        Exception: If initializing the database fails.
    """
    async with get_db_connection() as db:
        try:
            await migrate_schema()
            LoggingManager.info(
                "Database initialization and schema migration completed."
            )
        except Exception as e:
            LoggingManager.error(f"Failed to initialize database: {e}")
            raise


async def main():
    await load_database_configurations()
    # Additional initialization or operations can be added here


# Test suite implementation
async def run_tests():
    """
    Runs a suite of tests to verify the functionality of the database operations.

    This function performs a series of asynchronous tests to ensure the database operations work as expected. It covers tests for database connection, schema migration, image insertion and retrieval, and pagination parameter validation.
    """
    # Test database connection
    async with get_db_connection() as conn:
        assert conn is not None, "Database connection failed."

    # Test schema migration
    await migrate_schema()
    LoggingManager.info("Schema migration test passed.")

    # Test insert and retrieve compressed image
    test_hash = "test_hash"
    test_format = "png"
    test_image_path = (
        "/home/lloyd/EVIE/scripts/image_interconversion_gui/test_image.png"
    )
    with open(test_image_path, "rb") as image_file:
        test_data = image_file.read()

    await insert_compressed_image(test_hash, test_format, test_data)
    retrieved = await retrieve_compressed_image(test_hash)
    assert retrieved is not None, "Failed to retrieve compressed image."
    retrieved_data, retrieved_format = retrieved
    assert retrieved_format == test_format, "Retrieved format does not match."
    LoggingManager.info("Insert and retrieve compressed image test passed.")

    # Test get images metadata with pagination
    metadata = await get_images_metadata(0, 1)
    assert (
        isinstance(metadata, list) and len(metadata) <= 1
    ), "Get images metadata test failed."
    LoggingManager.info("Get images metadata test passed.")

    # Test validate pagination params
    try:
        validate_pagination_params(-1, 10)
    except ValueError:
        LoggingManager.info("Validate pagination params test passed.")
    else:
        assert False, "Validate pagination params test failed."

    LoggingManager.info("All tests passed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(run_tests())

# TODO:
# High Priority:
# - [ ] Investigate and mitigate potential security vulnerabilities related to image data handling. (Security)
# - [ ] Enhance error handling and provide more informative error messages. (Reliability)

# Medium Priority:
# - [ ] Explore the possibility of integrating a more advanced ORM for database interactions. (Maintainability)
# - [ ] Implement a more robust configuration management system, using YAML or JSON. (Maintainability)
# - [ ] Optimize database query performance, especially for large-scale image storage and retrieval. (Performance)

# Low Priority:
# - [ ] Explore the feasibility of adding support for alternative database engines like PostgreSQL, MySQL. (Scalability)
# - [ ] Consider adding more detailed metrics and monitoring around database operations.(Performance)

# Routine:
# - [ ] Refactor the codebase to improve readability and maintainability.
# - [ ] Update the documentation with additional examples and use cases.
# - [ ] Review and update the logging statements for consistency and clarity.
# - [ ] Implement unit tests for new functionalities and edge cases.
# - [ ] Review and update the `__all__` section to accurately reflect the module's public interface.
# - [ ] Ensure that all functions have appropriate type hints and docstrings.
# - [ ] Ensure that the code adheres to PEP 8 guidelines and formatting standards.
# - [ ] Perform a comprehensive review of the module for any potential improvements or optimizations.
# - [ ] Review the module for any potential security vulnerabilities or data leakage risks.

# Known Issues:
# - [ ] None
