import logging
import logging.config
import logging.handlers
import sys
import time
import asyncio
import aiofiles
from typing import (
    Dict,
    Any,
    Optional,
    TypeAlias,
    Callable,
    Awaitable,
    TypeVar,
    Coroutine,
    Union,
    Tuple,
    Type,
)
import pathlib
import json
from concurrent.futures import Executor, ThreadPoolExecutor
import functools
from functools import wraps
import tracemalloc
import inspect
from inspect import signature, Parameter
from indevalidate import AsyncValidationException, ValidationRules

T = TypeVar("T", bound=Callable[..., Awaitable[Any]])

# Module Header
"""
Module Name: indelogging.py
Description: This module provides a comprehensive and advanced logging setup for the INDEGO project development.
             It includes detailed configuration for various logging formats, handlers, and a custom asynchronous
             logging decorator to ensure non-blocking logging operations. It also ensures the logging configuration
             file exists or creates it with default settings.
Author: [Author Name]
Created Date: [Date]
Last Modified: [Date]
"""
__all__ = [
    "ensure_logging_config_exists",
    "configure_logging",
    "AsyncValidationException",
    "UniversalDecorator",
]

# Type Aliases
DIR_NAME: TypeAlias = str
DIR_PATH: TypeAlias = pathlib.Path
FILE_NAME: TypeAlias = str
FILE_PATH: TypeAlias = pathlib.Path
LogFunction: TypeAlias = Callable[..., Awaitable[None]]
true: TypeAlias = bool  # For Correct JSON Formatting
false: TypeAlias = bool  # For Correct JSON Formatting
T = TypeVar("T")
ValidationRules = Dict[str, Callable[[Any], Awaitable[bool]]]

# Define the type for the decorator that can handle both coroutine and regular functions.
Decorator = Callable[
    [Callable[..., Union[T, Coroutine[Any, Any, T]]]],
    Callable[..., Coroutine[Any, Any, T]],
]
# Constants
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(process)d - %(thread)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "%",
            "validate": true,
        },
        "verbose": {
            "format": "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(processName)s - %(threadName)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "%",
            "validate": true,
        },
        "ascii_art": {
            "format": "#########################################################\\n# %(asctime)s - %(levelname)s - %(module)s\\n# Function: %(funcName)s - Line: %(lineno)d\\n# Process: %(process)d - Thread: %(thread)d\\n# Message: \\n# %(message)s\\n#########################################################",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "%",
            "validate": true,
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "ascii_art",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "detailed.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "formatter": "ascii_art",
            "encoding": "utf-8",
            "delay": false,
        },
        "errors": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "errors_detailed.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "formatter": "verbose",
            "encoding": "utf-8",
            "delay": false,
        },
        "async_console": {
            "level": "DEBUG",
            "class": "concurrent_log_handler.ConcurrentRotatingFileHandler",
            "filename": "async_detailed.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "formatter": "ascii_art",
            "encoding": "utf-8",
            "delay": false,
        },
        "ascii_art_file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "ascii_art.log",
            "maxBytes": 52428800,
            "backupCount": 10,
            "formatter": "ascii_art",
            "encoding": "utf-8",
            "delay": false,
        },
    },
    "loggers": {
        "": {
            "handlers": [
                "console",
                "file",
                "errors",
                "async_console",
                "ascii_art_file",
            ],
            "level": "DEBUG",
            "propagate": true,
        },
        "async_logger": {
            "handlers": ["async_console"],
            "level": "DEBUG",
            "propagate": true,
        },
        "ascii_art_logger": {
            "handlers": ["ascii_art_file"],
            "level": "DEBUG",
            "propagate": true,
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file", "errors", "async_console", "ascii_art_file"],
    },
}

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


# Utility Functions
async def ensure_logging_config_exists(path: FILE_PATH) -> None:
    """
    Ensures that the logging configuration file exists at the specified path. If not, it creates the file
    with the default logging configuration.
    Args:
        path (FILE_PATH): The path to the logging configuration file.
    """
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, mode="w", encoding="utf-8") as config_file:
                json.dump(
                    DEFAULT_LOGGING_CONFIG, config_file, ensure_ascii=False, indent=4
                )
    except Exception as e:
        logging.error(f"Failed to ensure logging config exists: {e}", exc_info=True)


# Main Logging Configuration Setup
async def configure_logging() -> None:
    """
    Configures the logging system based on the logging configuration file. If the file does not exist,
    it ensures the file is created with the default configuration.
    """
    logging_conf_path: FILE_PATH = FILES["LOGGING_CONF"]
    await ensure_logging_config_exists(logging_conf_path)
    try:
        with logging_conf_path.open("r", encoding="utf-8") as config_file:
            logging_config = json.load(config_file)
            logging.config.dictConfig(logging_config)
    except Exception as e:
        logging.error(f"Failed to configure logging: {e}", exc_info=True)


class UniversalDecorator:
    """
    A highly sophisticated decorator designed to enhance the functionality of both synchronous and asynchronous functions.
    It incorporates advanced features such as automatic retry with exponential backoff, input validation based on custom rules,
    performance logging, and dynamic adjustment of retry strategies based on exceptions encountered during function execution.
    This decorator is meticulously crafted to ensure maximum flexibility, robustness, and efficiency in handling a wide range
    of use cases, making it an indispensable tool for any Python project.

    Attributes:
        retries (int): The maximum number of retries for the decorated function.
        delay (int): The initial delay between retries, which may be dynamically adjusted.
        log_config (Dict): The logging configuration to be used for logging within the decorator.
        validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Custom validation rules for function arguments.
        retry_exceptions (Tuple[Type[BaseException], ...]): Exceptions that trigger a retry.
        enable_performance_logging (bool): Flag to enable or disable performance logging.
        dynamic_retry_enabled (bool): Flag to enable or disable dynamic retry strategies.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: int = 2,
        log_config: Dict[str, Any] = DEFAULT_LOGGING_CONFIG,
        validation_rules: Optional[ValidationRules] = None,
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        enable_performance_logging: bool = True,
        dynamic_retry_enabled: bool = True,
    ) -> None:
        self.retries = retries
        self.delay = delay
        self.log_config = log_config
        self.validation_rules = validation_rules or {}
        self.retry_exceptions = retry_exceptions
        self.enable_performance_logging = enable_performance_logging
        self.dynamic_retry_enabled = dynamic_retry_enabled

    def __call__(
        self, func: Callable[..., Union[Awaitable[Any], Any]]
    ) -> Callable[..., Union[Awaitable[Any], Any]]:
        """
        Transforms the UniversalDecorator into a callable object, allowing it to be used as a decorator. This method
        dynamically determines whether the decorated function is synchronous or asynchronous and applies the appropriate
        wrapper to enhance its functionality with retries, validation, performance logging, and dynamic retry strategies.

        Args:
            func (Callable[..., Union[Awaitable[Any], Any]]): The function to be decorated.

        Returns:
            Callable[..., Union[Awaitable[Any], Any]]: The decorated function with enhanced functionality.
        """

        # Determine if the function is asynchronous using asyncio's built-in check.
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """
                An asynchronous wrapper function that provides the core functionality of the UniversalDecorator for asynchronous functions.
                It incorporates logic for validation, retrying with exponential backoff, performance logging, and dynamic retry strategy adjustment.

                Args:
                    *args: Positional arguments for the decorated function.
                    **kwargs: Keyword arguments for the decorated function.

                Returns:
                    Any: The result of the decorated function execution.
                """
                # Validate input arguments based on custom validation rules
                if self.validation_rules:
                    await self.validate_rules_async(func, *args, **kwargs)

                attempts = 0
                delay = self.delay
                while attempts < self.retries:
                    try:
                        start_time = time.perf_counter()
                        result = await func(*args, **kwargs)
                        end_time = time.perf_counter()
                        if self.enable_performance_logging:
                            await self.log_performance(
                                func.__name__, start_time, end_time
                            )
                        return result
                    except self.retry_exceptions as e:
                        attempts += 1
                        logging.error(
                            f"{func.__name__} attempt {attempts} failed with {e}, retrying after {delay} seconds..."
                        )
                        if self.dynamic_retry_enabled:
                            delay = await self.dynamic_retry_strategy(e, attempts)
                        await asyncio.sleep(delay)
                logging.debug(f"Final attempt for {func.__name__}")
                return await func(*args, **kwargs)

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """
                A synchronous wrapper function that directly executes the decorated synchronous function without event loop manipulation.
                This function retains core functionalities such as validation, retrying with exponential backoff, performance logging, and dynamic retry strategy adjustment,
                while simplifying the execution path for synchronous functions.

                Args:
                    *args: Positional arguments for the decorated function.
                    **kwargs: Keyword arguments for the decorated function.

                Returns:
                    Any: The result of the decorated function execution.
                """
                attempts = 0  # Initialize attempt counter
                delay = self.delay  # Initialize delay with the initial delay setting

                while (
                    attempts < self.retries
                ):  # Loop until the maximum number of retries is reached
                    try:
                        start_time = (
                            time.perf_counter()
                        )  # Record the start time for performance logging

                        # Validate input arguments based on custom validation rules
                        if self.validation_rules:
                            self.validate_rules_sync(func, *args, **kwargs)

                        result = func(
                            *args, **kwargs
                        )  # Directly execute the synchronous function

                        end_time = (
                            time.perf_counter()
                        )  # Record the end time for performance logging
                        if self.enable_performance_logging:
                            self.log_performance_sync(
                                func.__name__, start_time, end_time
                            )  # Log performance for synchronous functions

                        return result  # Return the result of the function execution

                    except (
                        self.retry_exceptions
                    ) as e:  # Catch specified exceptions for retry logic
                        attempts += 1  # Increment the attempt counter
                        logging.error(
                            f"{func.__name__} attempt {attempts} failed with {e}, retrying after {delay} seconds..."
                        )  # Log the error and retry attempt

                        if (
                            self.dynamic_retry_enabled
                        ):  # Check if dynamic retry strategy is enabled
                            delay = self.dynamic_retry_strategy_sync(
                                e, attempts
                            )  # Adjust delay based on dynamic retry strategy

                        time.sleep(
                            delay
                        )  # Wait for the specified delay before retrying

                logging.debug(
                    f"Final attempt for {func.__name__}"
                )  # Log the final attempt
                return func(*args, **kwargs)  # Execute the function one final time

            return sync_wrapper

    async def validate_rules_async(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> None:
        """
        Asynchronously validates the inputs to the decorated function based on the provided asynchronous validation rules.
        This method ensures that each argument passed to the function adheres to the predefined rules, enhancing the robustness
        and reliability of the function execution.

        Args:
            func (Callable[..., Awaitable[Any]]): The function being decorated.
            *args (Any): Positional arguments passed to the function.
            **kwargs (Any): Keyword arguments passed to the function.

        Raises:
            AsyncValidationException: If any argument fails to satisfy its corresponding asynchronous validation rule.
        """
        logging.debug(
            f"Validating async rules for function {func.__name__} with args {args} and kwargs {kwargs}"
        )
        if not hasattr(self, "_bound_arguments_checked"):
            bound_arguments = signature(func).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            self._bound_arguments_checked = True

            for arg, value in bound_arguments.arguments.items():
                if arg in self.validation_rules:
                    validation_rule = self.validation_rules[arg]
                    is_valid = (
                        await validation_rule(value)
                        if asyncio.iscoroutinefunction(validation_rule)
                        else validation_rule(value)
                    )
                    if not is_valid:
                        raise AsyncValidationException(
                            arg,
                            value,
                            f"Validation failed for argument '{arg}' with value '{value}'",
                        )

    async def log_performance_async(
        self, func_name: str, start_time: float, end_time: float
    ) -> None:
        """
        Logs the performance of the decorated function, including the time taken for execution.

        Args:
            func_name (str): The name of the function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.
        """
        adjusted_time = end_time - start_time
        logging.debug(f"{func_name} executed in {adjusted_time:.6f}s")

    async def dynamic_retry_strategy_async(
        self, exception: BaseException, attempt: int
    ) -> int:
        """
        Dynamically determines the retry delay based on the exception type and the number of attempts already made.
        This method allows for a more adaptive and responsive retry mechanism, potentially increasing the chances of success in subsequent attempts.

        Args:
            exception (BaseException): The exception that triggered the retry logic.
            attempt (int): The current retry attempt number.

        Returns:
            int: The delay in seconds before the next retry attempt.
        """
        if isinstance(exception, TimeoutError):
            return min(
                5, 2**attempt
            )  # Exponential backoff with a cap for timeout errors
        elif isinstance(exception, ConnectionError):
            return min(10, 2 * attempt)  # Linear backoff for connection errors
        return self.delay  # Default delay for all other exceptions

    # Synchronous Functions

    def log_performance_sync(
        self, func_name: str, start_time: float, end_time: float
    ) -> None:
        """
        Logs the performance of the decorated function, including the time taken for execution.

        Args:
            func_name (str): The name of the function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.
        """
        adjusted_time = end_time - start_time
        logging.debug(f"{func_name} executed in {adjusted_time:.6f}s")

    def dynamic_retry_strategy_sync(
        self, exception: BaseException, attempt: int
    ) -> int:
        """
        Dynamically determines the retry delay based on the exception type and the number of attempts already made.
        This method allows for a more adaptive and responsive retry mechanism, potentially increasing the chances of success in subsequent attempts.

        Args:
            exception (BaseException): The exception that triggered the retry logic.
            attempt (int): The current retry attempt number.

        Returns:
            int: The delay in seconds before the next retry attempt.
        """
        if isinstance(exception, TimeoutError):
            return min(
                5, 2**attempt
            )  # Exponential backoff with a cap for timeout errors
        elif isinstance(exception, ConnectionError):
            return min(10, 2 * attempt)  # Linear backoff for connection errors
        return self.delay  # Default delay for all other exceptions

    def validate_rules_sync(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> None:
        """
        Asynchronously validates the inputs to the decorated function based on the provided asynchronous validation rules.
        This method ensures that each argument passed to the function adheres to the predefined rules, enhancing the robustness
        and reliability of the function execution.

        Args:
            func (Callable[..., Awaitable[Any]]): The function being decorated.
            *args (Any): Positional arguments passed to the function.
            **kwargs (Any): Keyword arguments passed to the function.

        Raises:
            AsyncValidationException: If any argument fails to satisfy its corresponding asynchronous validation rule.
        """
        logging.debug(
            f"Validating async rules for function {func.__name__} with args {args} and kwargs {kwargs}"
        )
        if not hasattr(self, "_bound_arguments_checked"):
            bound_arguments = signature(func).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            self._bound_arguments_checked = True

            for arg, value in bound_arguments.arguments.items():
                if arg in self.validation_rules:
                    validation_rule = self.validation_rules[arg]
                    is_valid = (
                        asyncio.run(validation_rule(value))
                        if asyncio.iscoroutinefunction(validation_rule)
                        else validation_rule(value)
                    )
                    if not is_valid:
                        raise AsyncValidationException(
                            arg,
                            value,
                            f"Validation failed for argument '{arg}' with value '{value}'",
                        )


# Module Footer
"""
TODO:
- Investigate and integrate more advanced logging handlers and formatters for better log management.
- Explore the possibility of adding user-defined logging levels for more granular control over logging output.
- Implement a log rotation mechanism to manage log file sizes and prevent excessive disk space usage.
- Enhance the async_log_decorator to support configurable executors for better performance tuning.

Known Issues:
- None identified at this time.

Additional Functionalities:
- Future enhancements integration with indedatabase.py to store logs in a database for better log management.
- Future Enhancements: Integration with indecache.py for asynchronous smart caching of log messages.
- Future Enhancements: Integration with indeapi.py for logging API requests and responses for debugging purposes.
- Future Enhancements: Integration with indedecorators.py to contribute as a part of the decorator library for the INDEGO project.
- Future Enhancements: Integration with indeutils.py for additional utility functions and helper classes for logging operations.
"""
