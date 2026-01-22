"""
Image Interconversion GUI Database Module
=========================================

This module provides database functionalities essential for the Image Interconversion GUI application. It includes loading configurations, establishing database connections, executing queries, and managing image data with security measures like compression and encryption.

Dependencies:
- asyncio: For asynchronous programming.
- aiosqlite: Asynchronous SQLite database interaction.
- cryptography: For encryption and decryption of image data.
- PIL: For image processing tasks.

Setup/Initialization:
Ensure the following environment variables are set or a `config.ini` file is present:
- DATABASE_PATH: Path to the SQLite database file.
- KEY_FILE_PATH: Path to the encryption key file.

Author: Lloyd Handyside
Contact: ace1928@gmail.com
Version: 1.0.0
Creation Date: 2024-04-06
Last Modified: 2024-04-09
Last Reviewed: 2024-04-09
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
import unittest
from unittest.mock import patch, MagicMock

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
    Asynchronous context manager for managing database connections.

    This decorator ensures that the database connection is automatically opened at the start and properly closed after the block's execution, regardless of whether an exception occurred.

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
        parameters (tuple): Parameters for the SQL query.

    Returns:
        aiosqlite.Cursor: The cursor resulting from the query execution.

    Raises:
        aiosqlite.OperationalError: For operational database errors.
        Exception: For any other unexpected errors.
    """
    LoggingManager.debug(
        f"Executing database query: {query} with parameters: {parameters}"
    )
    try:
        async with get_db_connection() as db_connection:
            cursor = await db_connection.execute(query, parameters)
            await db_connection.commit()
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
    async with get_db_connection() as db_connection:
        cursor = await db_connection.execute(query, (limit, offset))
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
        async with get_db_connection() as db_connection:
            cursor = await db_connection.execute(query, (hash,))
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

    async with get_db_connection() as db_connection:
        for table, expected_columns in expected_schema.items():
            cursor = await db_connection.execute(f"PRAGMA table_info({table})")
            current_columns = {info[1]: info[2] for info in await cursor.fetchall()}
            missing_columns = [
                col for col in expected_columns if col[0] not in current_columns
            ]

            for col_name, col_type in missing_columns:
                alter_query = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                await db_connection.execute(alter_query)
                LoggingManager.info(
                    f"Added missing column '{col_name}' to '{table}' table."
                )

        await db_connection.commit()
        LoggingManager.info("Database schema migration completed successfully.")


async def init_db():
    """
    Initializes the database and migrates the schema to the latest version.

    This function ensures that the database schema is up to date by calling the migrate_schema function. It is intended to be run at application startup to prepare the database for use.

    Raises:
        Exception: If initializing the database fails.
    """
    async with get_db_connection() as db_connection:
        try:
            await migrate_schema()
            LoggingManager.info(
                "Database initialization and schema migration completed."
            )
        except Exception as e:
            LoggingManager.error(f"Failed to initialize database: {e}")
            raise


class TestDatabaseOperations(unittest.TestCase):
    @patch("database.ConfigManager")
    async def test_load_database_configurations(self, mock_config_manager):
        mock_config_manager_instance = mock_config_manager.return_value
        mock_config_manager_instance.load_config = MagicMock()
        mock_config_manager_instance.get = MagicMock(
            side_effect=["test_db.sqlite", "test_key.key"]
        )
        await load_database_configurations()
        mock_config_manager_instance.load_config.assert_called_once()
        self.assertEqual(DatabaseConfig.DB_PATH, "test_db.sqlite")
        self.assertEqual(DatabaseConfig.KEY_FILE, "test_key.key")

    @patch("database.aiosqlite.connect")
    async def test_get_db_connection(self, mock_connect):
        mock_connect.return_value = asyncio.Future()
        mock_connect.return_value.set_result(get_db_connection(":memory:"))
        async with get_db_connection() as connection:
            self.assertIsInstance(get_db_connection, connection)

    @patch("database.get_db_connection")
    async def test_execute_db_query(self, mock_get_db_connection):
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall = MagicMock(return_value=asyncio.Future())
        mock_cursor.fetchall.return_value.set_result([("test_hash", "test_format")])
        mock_connection.execute = MagicMock(return_value=mock_cursor)
        mock_get_db_connection.return_value.__aenter__.return_value = mock_connection
        result = await execute_db_query("SELECT * FROM images")
        self.assertEqual(await result.fetchall(), [("test_hash", "test_format")])

    def test_validate_pagination_params(self):
        with self.assertRaises(ValueError):
            validate_pagination_params(-1, 10)
        with self.assertRaises(ValueError):
            validate_pagination_params(0, 0)
        # No exception should be raised for valid parameters
        validate_pagination_params(0, 10)

    @patch("database.execute_db_query")
    async def test_get_images_metadata(self, mock_execute_db_query):
        mock_execute_db_query.return_value = asyncio.Future()
        mock_execute_db_query.return_value.set_result([("test_hash", "test_format")])
        result = await get_images_metadata(0, 1)
        self.assertEqual(result, [("test_hash", "test_format")])

    @patch("database.execute_db_query")
    async def test_insert_compressed_image(self, mock_execute_db_query):
        await insert_compressed_image("test_hash", "test_format", b"test_data")
        mock_execute_db_query.assert_called_once()

    @patch("database.execute_db_query")
    async def test_retrieve_compressed_image(self, mock_execute_db_query):
        mock_execute_db_query.return_value = asyncio.Future()
        mock_execute_db_query.return_value.set_result([("test_data", "test_format")])
        result = await retrieve_compressed_image("test_hash")
        self.assertEqual(result, ("test_data", "test_format"))

    @patch("database.get_db_connection")
    async def test_migrate_schema(self, mock_get_db_connection):
        mock_connection = MagicMock()
        mock_get_db_connection.return_value.__aenter__.return_value = mock_connection
        await migrate_schema()
        mock_connection.execute.assert_called()

    @patch("database.migrate_schema")
    async def test_init_db(self, mock_migrate_schema):
        await init_db()
        mock_migrate_schema.assert_called_once()


if __name__ == "__main__":
    asyncio.run(unittest.main())

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
