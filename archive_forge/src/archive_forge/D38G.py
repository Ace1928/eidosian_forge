import hashlib
import os
import asyncio
import aiofiles
from inspect import getsource, getmembers, isfunction, isclass
from Cython.Build import cythonize
from anyio import Path
from setuptools import setup, Extension
import sys
import logging
import logging.config
import pathlib
from typing import (
    Any,
    Callable,
    Coroutine,
    Type,
    Dict,
    List,
    Tuple,
    TypeAlias,
    Union,
    Optional,
)
from indelogging import (
    DEFAULT_LOGGING_CONFIG,
    ensure_logging_config_exists,
    async_log_decorator,
)

# Comprehensive Type Aliases for Enhanced Code Readability and Consistency
# The following section defines a series of type aliases aimed at improving the readability and maintainability of the codebase. By providing explicit, descriptive types for commonly used concepts such as directory names, paths, file names, and file paths, these aliases facilitate a more intuitive and error-resistant development process. They leverage Python's typing system to ensure consistent usage of these types across the codebase, thereby enhancing the robustness and clarity of the code.

# Type alias for directory names. Expected to be string values representing the name of a directory.
DIR_NAME: TypeAlias = str
# Type alias for directory paths. Utilizes the pathlib.Path type for a clear, platform-independent representation of filesystem paths.
DIR_PATH: TypeAlias = pathlib.Path
# Type alias for file names. Similar to DIR_NAME, it represents file names as strings, aiding in distinguishing between general strings and file names.
FILE_NAME: TypeAlias = str
# Type alias for file paths. Employs pathlib.Path to represent paths to files, helping differentiate between directory paths (DIR_PATH) and file paths (FILE_PATH).
FILE_PATH: TypeAlias = pathlib.Path

# Immutable Codebase-Wide Constants
# The following dictionaries define constants for directories and files used throughout the codebase. These constants are not to be modified under any circumstances to ensure the integrity and consistency of file and directory references. They serve as a centralized point of reference for all directory and file paths, streamlining path management and reducing the risk of errors.

# Dictionary mapping directory names to their respective paths. Optional typing allows for the possibility of uninitialized paths.
DIRECTORIES: Dict[DIR_NAME, Optional[DIR_PATH]] = {
    "ROOT": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development"),
    "LOGS": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/logs"),
    "CONFIG": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config"
    ),
    "DATA": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/data"),
    "MEDIA": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/media"),
    "SCRIPTS": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/scripts"
    ),
    "TEMPLATES": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/templates"
    ),
    "UTILS": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/utils"),
}
# Dictionary mapping file names to their respective paths, ensuring that file paths are correctly typed and managed.
FILES: Dict[FILE_NAME, FILE_PATH] = {
    "DIRECTORIES_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/directories.conf"
    ),
    "FILES_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/files.conf"
    ),
    "DATABASE_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/database.conf"
    ),
    "API_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/api.conf"
    ),
    "CACHE_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/cache.conf"
    ),
    "LOGGING_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/logging.conf"
    ),
}
# Enhanced logging configuration for detailed output and debugging
# Define a comprehensive logging configuration dictionary
# Correctly accessing the 'LOGGING_CONF' file path from the FILES dictionary
logging_conf_path: FILE_PATH = FILES["LOGGING_CONF"]

# Ensure the logging configuration file exists before proceeding
logging_setup = asyncio.run(ensure_logging_config_exists(logging_conf_path))

# Since the logging configuration is now guaranteed to exist, we can proceed to use it as intended
logging_config: Dict[str, Any] = DEFAULT_LOGGING_CONFIG
# Apply the logging configuration to initialize the logging system according to the defined specifications.
logging.config.dictConfig(logging_config)


class CythonCompiler:
    """
    Manages the compilation of Python code into Cython C extensions, ensuring that the C version is
    up-to-date with its Python counterpart, and handles dynamic execution based on code integrity.
    """

    @classmethod
    async def create(cls, obj: Type) -> "CythonCompiler":
        """
        Creates a new instance of the CythonCompiler class asynchronously.
        Args:
            obj (Type): The Python object to be compiled into a Cython C extension.

        Returns:
            CythonCompiler: A new instance of the CythonCompiler class.
        """
        source_code: str = (
            await cls._extract_source_code(obj)
            or ""  # Corrected parameter from CythonCompiler to obj
        )  # Ensure source_code is a string
        return cls(obj, source_code)

    def __init__(self, obj: Type, source_code: str) -> None:
        """
        Initializes the CythonCompiler with a Python object, typically a function or class.

        Args:
            obj (Type): The Python object to be compiled into a Cython C extension.
        """
        self.obj: Type = obj
        self.module_name: str = f"{obj.__name__}_{self._generate_hash()}"
        self.source_code: str = source_code
        self.pyx_file: str = f"{self.module_name}.pyx"
        self.hash_file: str = f"{self.module_name}.hash"

    @async_log_decorator
    async def _extract_source_code(self, obj: Type) -> str:
        """
        Extracts and concatenates the source code of a Python object and all nested members.

        Args:
            obj (Type): The Python object whose source code is to be extracted.

        Returns:
            str: The concatenated source code of the object and its members.
        """
        source: str = ""
        try:
            source = getsource(obj)
            members = getmembers(obj, predicate=lambda x: isfunction(x) or isclass(x))
            for _, member in members:
                if member is not obj:
                    source += await self._extract_source_code(member)
            return source
        except Exception as e:
            logger.error(f"Failed to extract source code: {e}")
            return source  # Ensures a string is always returned, even in error cases.

    @async_log_decorator
    def _generate_hash(self) -> str:
        """
        Generates a SHA-256 hash of the concatenated source code for integrity checking.

        Returns:
            str: The SHA-256 hash of the source code.
        """
        try:
            return hashlib.sha256(self.source_code.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate hash: {e}")
            return ""

    @async_log_decorator
    async def _file_exists_and_matches(
        self, filename: str, expected_content: str
    ) -> bool:
        """
        Checks if a file exists and its content matches the expected content.

        Args:
            filename (str): The name of the file to check.
            expected_content (str): The expected content to match against the file's content.

        Returns:
            bool: True if the file exists and its content matches the expected content, False otherwise.
        """
        try:
            if os.path.exists(filename):
                async with aiofiles.open(filename, "r") as file:
                    content = await file.read()
                return content == expected_content
            return False
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return False

    @async_log_decorator
    async def _write_to_file(self, filename: str, content: str) -> None:
        """
        Asynchronously writes content to a file.

        Args:
            filename (str): The name of the file to write to.
            content (str): The content to write to the file.
        """
        try:
            async with aiofiles.open(filename, "w") as file:
                await file.write(content)
        except Exception as e:
            logger.error(f"Failed to write to file {filename}: {e}")

    @async_log_decorator
    def _compile_cython(self) -> None:
        """
        Compiles the Python source to a Cython C extension.
        """
        try:
            with open(self.pyx_file, "w") as file:
                file.write(self.source_code)
            setup(
                ext_modules=cythonize(
                    Extension(self.module_name, sources=[self.pyx_file])
                ),
                script_args=["build_ext", "--inplace"],
            )
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
        finally:
            sys.argv = sys.argv[:1]

    @async_log_decorator
    def _execute_c(self) -> None:
        """
        Dynamically imports and executes the compiled C extension.
        """
        try:
            __import__(self.module_name)
        except ImportError as e:
            logger.error(f"Failed to import module {self.module_name}: {e}")

    @async_log_decorator
    async def ensure_latest_version_and_execute(self) -> None:
        """
        Ensures the latest version of the Python object is compiled to C and executes it.
        """
        current_hash = self._generate_hash()
        if await self._file_exists_and_matches(self.hash_file, current_hash):
            logger.info(
                f"No changes detected. Executing C version of {self.obj.__name__}."
            )
            self._execute_c()
        else:
            logger.info(
                f"Changes detected or C version not found. Updating {self.obj.__name__}."
            )
            self._compile_cython()
            await self._write_to_file(self.hash_file, current_hash)
            self._execute_c()

    @async_log_decorator
    def convert_to_c_and_run(self) -> None:
        """
        Converts the Python object to a Cython C extension and ensures its execution is based on the latest code version.
        """
        asyncio.run(self.ensure_latest_version_and_execute())


@async_log_decorator
def cythonize_function(obj: Type) -> Type:
    """
    Decorator to automatically convert a function or class to a Cython C extension.

    Args:
        obj (Type): The Python object to be converted and executed as a Cython C extension.

    Returns:
        Type: The original Python object, unmodified.
    """
    compiler = CythonCompiler(obj)
    compiler.convert_to_c_and_run()
    return obj


# Example function decorated to demonstrate the functionality
@async_log_decorator
@cythonize_function
def example_function() -> None:
    print("Hello, world in C!")


if __name__ == "__main__":
    example_function()

"""
TODO List for Further Refinements:
- Maximize multiprocessing and asynchronous methods in all possible aspects for parallelized operation of maximum efficacy and efficiency.
- Optimize Source Code Extraction: Enhance the recursive extraction logic to handle more complex structures and edge cases.
- Improve Error Handling: Expand error handling to cover more specific exceptions, particularly during file operations and dynamic imports.
- Implement a caching mechanism for compiled C extensions to avoid recompilation of unchanged code, enhancing efficiency.
- Explore the integration of more advanced Cython features to further optimize the compiled C extensions.
- Develop a more robust system for managing and cleaning up generated files (.pyx, .c, .so) to prevent clutter.
- Investigate the possibility of integrating with other Python-to-C compilers or tools for broader compatibility and optimization options.
"""
