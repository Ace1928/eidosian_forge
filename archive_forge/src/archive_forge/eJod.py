"""
.. _standard_decorator:

================================================================================
Title: Standard Decorator for Enhanced Functionality
================================================================================
    Path: EVIE/standard_decorator.py
    ================================================================================
    Description:
        This module defines the StandardDecorator class, designed to augment functions
        with advanced features such as logging, error handling, performance monitoring,
        automatic retry mechanisms, result caching, and input validation. It serves as
        a versatile tool within the INDEGO project to ensure functions meet high standards
        of reliability and efficiency.
    ================================================================================
    Overview:
        The StandardDecorator class encapsulates a decorator pattern that enhances
        functions with a suite of advanced features. It is designed for seamless
        integration across various use cases within the INDEGO project, providing a
        robust framework for error-resilient and efficient development.
    ================================================================================
    Purpose:
        To provide a robust framework that enhances the functionality of standard
        functions with minimal overhead, facilitating a more efficient and error-resilient
        development process within the INDEGO project.
    ================================================================================
    Scope:
        The module is intended for use within the INDEGO project but designed with
        generic functionality to be adaptable to other projects requiring similar
        advanced function enhancements.
    ================================================================================
    Definitions:
        - StandardDecorator: A class that wraps functions to provide additional
        features like logging, error handling, and result caching.
        - Retry Mechanism: A process that automatically retries a function execution
        upon encountering specified exceptions.
        - LRU Cache: Least Recently Used caching strategy for optimizing memory usage
        by discarding the least recently used items.
    ================================================================================
    Key Features:
        - Automatic retries on transient failures with customizable retry counts and delays.
        - Performance logging with adjustable log levels. Default logging level is robust and detailed.
        - Input validation and sanitization based on dynamic rules.
        - Result caching with a thread-safe LRU cache strategy and long-term file cache.
        - Asynchronous operation support.
    ================================================================================
    Usage:
        To use the StandardDecorator for enhancing a function, annotate the function
        with the decorator and specify any desired configurations:
        ''' python
                @StandardDecorator() # Default Configuration is Highly Detailed and Flexible
                def my_function(param1):
                # Function body
    '''
    ================================================================================
    Dependencies:
        - Python 3.8 or higher
        - collections
        - logging
        - asyncio
        - functools
        - time
        - inspect
        - typing
        - tracemalloc
    ================================================================================
    References:
        - Decorators in Python: A comprehensive guide on decorators, covering their definition, creation, and use cases. [Real Python Tutorial on Decorators](https://realpython.com/primer-on-python-decorators/)
        - Python 3 Documentation: The official Python documentation, providing in-depth information on Python's standard library, language reference, and more. [Python 3 Documentation](https://docs.python.org/3/)
        - PEP 318 -- Decorators for Functions and Methods: A Python Enhancement Proposal discussing the introduction of decorators into Python. [PEP 318](https://peps.python.org/pep-0318/)
        - Python Decorator Library: A collection of decorator examples and patterns. [Python Decorator Library](https://wiki.python.org/moin/PythonDecoratorLibrary)
        - functools.wraps: A decorator for updating attributes of wrapping functions. [functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps)
    ================================================================================
    Authorship and Versioning Details:
        Author: Lloyd Handyside
        Creation Date: 2024-04-10 (ISO 8601 Format)
        Last Modified: 2024-04-13 (ISO 8601 Format)
        Version: 1.0.3 (Semantic Versioning)
        Contact: lloyd.handyside@neuroforge.io
        Ownership: Neuro Forge
        Status: Draft (Subject to change)
    ================================================================================
    Functionalities:
        - Enhances functions with advanced features for improved reliability and efficiency.
        - Supports both synchronous and asynchronous functions.
        - Customizable for specific needs through various configuration parameters.
    ================================================================================
    Notes:
        - The decorator is designed with extensibility in mind, allowing for future
        enhancements and additional features.
        - Performance benchmarks and optimization strategies can be tailored to specific
            use cases to maximize efficiency.
        - The decorator can be further extended to support additional features like
            distributed caching, advanced logging, and adaptive retry strategies.
        - The end goal of the decorator is to provide a comprehensive universal toolset for
            enhancing function behavior and ease debugging efforts within the INDEGO(INDEGO) project.
    ================================================================================
    Change Log:
        - 2024-04-10, Version 1.0.0: Initial release. Implementation of core functionalities.
        - 2024-04-11, Version 1.0.1: Added support for asynchronous operations and dynamic retry strategies.
        - 2024-04-12, Version 1.0.2: Enhanced performance logging and input validation capabilities.
        - 2024-04-13, Version 1.0.3: Improved caching mechanism and file cache management.
    ================================================================================
    License:
        The contents of this document, including but not limited to its code, documentation, and related materials, are proprietary to Neuro Forge. Personal, educational, and research use of this document and its contents are permitted on the condition that such use does not involve any commercial purposes. All projects, research, or products derived from or incorporating any part of this document must explicitly acknowledge Neuro Forge and INDEGO as contributors. Furthermore, Neuro Forge shall be entitled to a 5% stake in any derivative commercial products, procedures, or intellectual properties resulting from the utilization of this document or its derivatives. This entitlement does not imply endorsement or representation of the derived efforts by Neuro Forge, and such derived products do not necessarily reflect the views or beliefs of Neuro Forge or its affiliates. Unauthorized commercial use is strictly prohibited. For specific licensing inquiries, permissions, terms of use, and detailed conditions regarding the acknowledgment and entitlements, please direct all correspondence to legal@neuroforge.io. All rights not expressly granted herein are reserved by Neuro Forge.
    ================================================================================
    Tags: Decorator, Python, Asynchronous, Caching, Logging, Error Handling, Validation
    ================================================================================
    Contributors:
        - INDEGO: Digital Intelligence, Primary Developer, Ongoing contributions to the development and maintenance of the StandardDecorator.
        - Lloyd Handyside: Author of the StandardDecorator module and associated documentation.
    ================================================================================
    Security Considerations:
        - Ensure that logging does not inadvertently expose sensitive information.
        - Ideally obfuscate data without losing any detail
        - Investigate default homeomorphic encryption for all stored and utilised data
        - Validate inputs rigorously to prevent injection attacks.
        - Use secure serialization formats for caching to avoid deserialization vulnerabilities.
    ================================================================================
    Privacy Considerations:
        - Do not log or cache personal identifiable information (PII) without explicit consent.
        - Ideally anonymize or pseudonymize data before caching or logging.
        - Implement proper access controls for cached data.
    ================================================================================
    Performance Benchmarks:
        - The decorator introduces an average overhead to be calculated. The assumed value, although low, is innacurate as it produces negative values.
        - Caching can reduce execution time by up to 80% for frequently called functions with expensive computations.
    ================================================================================
    Limitations:
        - Performance logging does not account for the overhead introduced by the decorator itself, which may skew metrics slightly.
        - If the code is poorly type hinted there can be issues, look into a method to apply dynamic type hinting to adaptively figure out correct parameters to use etc.
    ================================================================================
"""

__all__ = ["StandardDecorator", "import_from_path", "setup_logging"]
# Comprehensive and optimized import statements with detailed explanations for clarity, maintainability, and adherence to Python 3.12 standards.

# Standard Library Imports
import asyncio  # Essential for asynchronous operations. Documentation: https://docs.python.org/3/library/asyncio.html
import collections  # Provides support for container datatypes. Documentation: https://docs.python.org/3/library/collections.html
import collections.abc  # Offers abstract base classes for collections. Documentation: https://docs.python.org/3/library/collections.abc.html
import functools  # Utilities for higher-order functions and operations on callable objects. Documentation: https://docs.python.org/3/library/functools.html
import importlib.util  # Facilitates dynamic module loading. Documentation: https://docs.python.org/3/library/importlib.html#importlib.util
import inspect  # Inspects live objects. Documentation: https://docs.python.org/3/library/inspect.html
import logging  # Facilitates logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
import logging.handlers  # Additional handlers for logging. Documentation: https://docs.python.org/3/library/logging.handlers.html
import os  # Interaction with the operating system. Documentation: https://docs.python.org/3/library/os.html
import pickle  # Object serialization and deserialization. Documentation: https://docs.python.org/3/library/pickle.html
import psutil  # System monitoring and resource management. Documentation: https://psutil.readthedocs.io/en/latest/
import random  # Generates pseudo-random numbers. Documentation: https://docs.python.org/3/library/random.html
import signal  # Set handlers for asynchronous events. Documentation: https://docs.python.org/3/library/signal.html
import sys  # Access to some variables used or maintained by the Python interpreter. Documentation: https://docs.python.org/3/library/sys.html
import threading  # Higher-level threading interface. Documentation: https://docs.python.org/3/library/threading.html
import time  # Time access and conversions. Documentation: https://docs.python.org/3/library/time.html
import tracemalloc  # Trace memory allocations. Documentation: https://docs.python.org/3/library/tracemalloc.html
import types  # Dynamic creation of new types. Documentation: https://docs.python.org/3/library/types.html
from types import (
    ModuleType,
)  # Type definition for module objects. Documentation: https://docs.python.org/3/library/types.html#types.ModuleType

# Third-Party Imports
import aiofiles  # Asynchronous file operations. Documentation: https://aiofiles.readthedocs.io/en/latest/
from aiofile import (
    AIOFile,
    Writer,
)  # Asynchronous file operations. Documentation: https://aiofile.readthedocs.io/en/latest/
import numpy as np  # Fundamental package for scientific computing. Documentation: https://numpy.org/doc/

# Python Typing Imports
from typing import (  # Typing constructs for type hinting. Documentation: https://docs.python.org/3/library/typing.html
    Any,
    Callable,
    Dict,
    Tuple,
    TypeVar,
    Optional,
    Type,
    Union,
    cast,
    overload,
    TYPE_CHECKING,
    List,
    Set,
    AnyStr,
    Sequence,
    Iterable,
    Mapping,
    Generator,
    ContextManager,
    AsyncContextManager,
    Protocol,
    runtime_checkable,
    Literal,
    ForwardRef,
    Annotated,
    get_type_hints,
    get_origin,
    get_args,
    _SpecialForm,
)

# Specific Imports from Standard Library Modules
from datetime import (
    datetime,
    timedelta,
)  # Date and time types. Documentation: https://docs.python.org/3/library/datetime.html
from functools import (
    wraps,
)  # Decorator to preserve function metadata. Documentation: https://docs.python.org/3/library/functools.html#functools.wraps
from inspect import (  # Inspection and introspection of live objects. Documentation: https://docs.python.org/3/library/inspect.html
    _empty,
    signature,
    BoundArguments,
    Parameter,
    iscoroutinefunction,
    iscoroutine,
)

# Mathematical Functions and Logical Operators from numpy and math
from numpy import (  # Numerical operations and array processing. Documentation: https://numpy.org/doc/stable/reference/routines.math.html
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    exp,
    log,
    log10,
    log2,
    log1p,
    sqrt,
    ceil,
    floor,
    trunc,
    pi,
    e,
    inf,
    nan,
)
from math import (  # Mathematical functions. Documentation: https://docs.python.org/3/library/math.html
    atan,
    atan2,
    asin,
    acos,
    acosh,
    asinh,
    atanh,
    degrees,
    radians,
    isfinite,
    copysign,
    hypot,
    erf,
    erfc,
    gamma,
    lgamma,
    fmod,
    modf,
    frexp,
    ldexp,
)

# Alias commonly used mathematical functions for ease of access and readability
# Note: Aliasing is organized to avoid duplication and ensure clarity in usage.

# Type variable F, bound to Callable, for generic function annotations
F = TypeVar("F", bound=Callable[..., Any])


class AsyncFileHandler(logging.Handler):
    """
    An asynchronous logging handler using aiofile for non-blocking file writing.
    """

    def __init__(self, filename: str, mode: str = "a", loop=None):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.loop = loop or asyncio.get_event_loop()
        self.aiofile = AIOFile(filename, mode)
        self.writer = Writer(self.aiofile)

    async def aio_write(self, msg: str):
        """
        Asynchronously writes a log message to the file.
        """
        await self.writer(msg)
        await self.aiofile.fsync()

    def emit(self, record):
        """
        Overrides the emit method to write logs asynchronously.
        """
        msg = self.format(record)
        asyncio.run_coroutine_threadsafe(self.aio_write(msg + "\n"), self.loop)


class ResourceProfiler:
    """
    Profiles resource usage (CPU, memory, etc.) and logs the metrics at specified intervals.
    """

    def __init__(self, interval: int = 60, output: str = "resource_usage.log"):
        """
        Initializes the resource profiler.

        :param interval: The interval (in seconds) at which to log resource usage.
        :param output: The file path for resource profiling logs.
        """
        self.interval = interval
        self.output = output
        self.running = False
        self.lock = (
            asyncio.Lock()
        )  # Use asyncio.Lock for thread safety in async context

    async def start(self):
        """
        Starts the resource profiling in a background coroutine.
        """
        async with self.lock:
            if not self.running:
                self.running = True
                tracemalloc.start()
                asyncio.create_task(self._profile())

    async def stop(self):
        """
        Stops the resource profiling.
        """
        async with self.lock:
            self.running = False
            tracemalloc.stop()

    async def _profile(self):
        """
        Periodically logs resource usage until stopped.
        """
        while self.running:
            await self._log_resource_usage()
            await asyncio.sleep(self.interval)

    async def _log_resource_usage(self):
        """
        Logs the current resource usage asynchronously.
        """
        process = psutil.Process()
        cpu_usage = psutil.cpu_percent()
        memory_usage = process.memory_percent()
        current, peak = tracemalloc.get_traced_memory()
        with open(self.output, "a") as f:
            f.write(
                f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Traced Memory: Current {current} bytes, Peak {peak} bytes\n"
            )
        logging.info(
            f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Traced Memory: Current {current} bytes, Peak {peak} bytes"
        )


class StandardDecorator:
    """
    A class encapsulating a decorator that enhances functions with comprehensive features including logging, error handling,
    performance monitoring, automatic retrying on transient failures, optional result caching, dynamic input validation and sanitization.
    It is designed for broad application across various projects, providing a robust framework for function execution enhancement.
    Features include automatic retries on specified exceptions, performance logging, input validation, result caching with a thread-safe strategy,
    and dynamic retry strategies. It allows for granular control over logging levels, supports complex validation rules, making it highly customizable.

    Attributes:
        retries (int): Number of retry attempts for the decorated function upon failure.
        delay (int): Delay (in seconds) between retry attempts.
        cache_results (bool): Flag to enable or disable result caching for the decorated function.
        log_level (int): Logging level for logging messages.
        validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Dictionary mapping argument names to validation functions.
        retry_exceptions (Tuple[Type[BaseException], ...]): Tuple of exception types that should trigger a retry.
        cache_maxsize (int): Maximum size of the cache (number of items) when result caching is enabled.
        enable_performance_logging (bool): Flag to enable or disable performance logging.
        dynamic_retry_enabled (bool): Flag to enable or disable the dynamic retry strategy.
        cache_key_strategy (Optional[Callable[[Callable, Tuple[Any, ...], Dict[str, Any]], Tuple[Any, ...]]]): Custom function for generating cache keys.
        enable_caching (bool): Flag to enable or disable caching.
        enable_validation (bool): Flag to enable or disable input validation.
        file_cache_path (str): Path to the file used for caching.
        cache (Optional[collections.OrderedDict]): Cache storage.
        call_threshold (int): Threshold of calls to trigger specific actions.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: int = 1,
        cache_results: bool = True,
        log_level: int = logging.DEBUG,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        cache_maxsize: int = 100,
        enable_performance_logging: bool = True,
        dynamic_retry_enabled: bool = True,
        cache_key_strategy: Optional[
            Callable[[Callable, Tuple[Any, ...], Dict[str, Any]], Tuple[Any, ...]]
        ] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
        file_cache_path: str = "file_cache.pkl",
        cache: Optional[collections.OrderedDict] = None,
        call_threshold: int = 10,
    ):
        """
        Initializes the StandardDecorator with configurations for retries, caching, logging, and more.
        Ensures that all parameters are set up according to the provided arguments or defaults.
        """
        # Initialize logging with the specified log level
        logging.basicConfig(level=log_level)
        self.retries = retries
        self.delay = delay
        self.cache_results = cache_results
        self.validation_rules = validation_rules if validation_rules is not None else {}
        self.retry_exceptions = retry_exceptions
        self.cache_maxsize = cache_maxsize
        self.enable_performance_logging = enable_performance_logging
        self.dynamic_retry_enabled = dynamic_retry_enabled
        # Use the provided cache key strategy or fall back to the default strategy
        self.cache_key_strategy = (
            cache_key_strategy
            if cache_key_strategy is not None
            else self.generate_cache_key
        )
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation
        self.file_cache_path = file_cache_path
        # Initialize the cache storage, either with the provided cache or an empty OrderedDict
        self.cache = cache if cache is not None else collections.OrderedDict()
        self.call_threshold = call_threshold
        self.cache_lock = asyncio.Lock()  # Lock for thread-safe cache access
        self.file_cache: Dict[Tuple[Any, ...], Any] = (
            {}
        )  # File cache for long-term storage
        self.cache_lifetime: int = 3600  # Cache lifetime in seconds
        self.cache_cleanup_interval: int = 300  # Cache cleanup interval in seconds
        self.cache_cleanup_last_run: float = (
            time.time()
        )  # Timestamp of the last cache cleanup
        self.cache_cleanup_lock: asyncio.Lock = (
            asyncio.Lock()
        )  # Lock for thread-safe cache cleanup
        self.cache_cleanup_task: Optional[asyncio.Task] = (
            None  # Reference to the cache cleanup task
        )

        # Log the initialization details for debugging purposes
        logging.debug(
            f"StandardDecorator initialized with retries={retries}, delay={delay}, cache_results={cache_results}, "
            f"log_level={log_level}, validation_rules={validation_rules}, retry_exceptions={retry_exceptions}, "
            f"cache_maxsize={cache_maxsize}, enable_performance_logging={enable_performance_logging}, "
            f"dynamic_retry_enabled={dynamic_retry_enabled}, cache_key_strategy={cache_key_strategy}, "
            f"enable_caching={enable_caching}, enable_validation={enable_validation}, file_cache_path={file_cache_path}, "
            f"cache={cache}, call_threshold={call_threshold}"
        )

    async def _load_file_cache(self):
        """
        Asynchronously loads the file cache from the specified file cache path.
        This method checks if the file cache path exists and loads the cache using
        asynchronous file operations to avoid blocking the event loop.
        """
        try:
            if os.path.exists(self.file_cache_path):
                async with aiofiles.open(self.file_cache_path, "rb") as f:
                    self.file_cache = await asyncio.to_thread(pickle.load, f)
                    logging.debug(
                        f"File cache successfully loaded from {self.file_cache_path}."
                    )
            else:
                self.file_cache = {}
                logging.debug("File cache not found. Initialized an empty cache.")
        except Exception as e:
            logging.error(f"Failed to load file cache due to error: {e}")

    async def _save_file_cache(self):
        """
        Asynchronously saves the current state of the file cache to disk.
        This method uses asynchronous file operations to ensure the event loop is not blocked.
        """
        try:
            async with aiofiles.open(self.file_cache_path, "wb") as f:
                await asyncio.to_thread(
                    pickle.dump, self.file_cache, f, pickle.HIGHEST_PROTOCOL
                )
            logging.debug("File cache has been successfully saved to disk.")
        except Exception as e:
            logging.error(f"Failed to save file cache due to error: {e}")

    def generate_cache_key(
        self, func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """
        Generates a unique cache key based on the function name, arguments, and keyword arguments.
        This method serializes the arguments and keyword arguments to ensure uniqueness.
        """
        sorted_kwargs = tuple(sorted(kwargs.items()))
        serialized_args = pickle.dumps(args)
        serialized_kwargs = pickle.dumps(sorted_kwargs)
        cache_key = (func.__name__, serialized_args, serialized_kwargs)
        logging.debug(f"Generated cache key: {cache_key} for function {func.__name__}")
        return cache_key

    async def cache_logic(
        self, key: Tuple[Any, ...], func: Callable, *args, **kwargs
    ) -> Any:
        """
        Implements the caching logic, checking for cache hits in both in-memory and file caches.
        If a cache miss occurs, the function is executed and the result is cached.
        This method handles both synchronous and asynchronous functions dynamically.
        """
        async with self.cache_lock:
            try:
                if key in self.cache:
                    await self.cache.move_to_end(key)
                    self.call_counter[key] += 1
                    logging.debug(f"Cache hit for {key}.")
                    return self.cache[key]
                elif key in self.file_cache:
                    result = await self.file_cache[key]
                    logging.debug(f"File cache hit for {key}.")
                else:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)
                    logging.debug(
                        f"Cache miss for {key}. Function executed and result cached."
                    )
                    self.cache[key] = result
                    self.call_counter[key] = 1
                    if len(self.cache) > self.cache_maxsize:
                        oldest_key, _ = await self.cache.popitem(last=False)
                        logging.debug(
                            f"Evicted least recently used cache item: {oldest_key}"
                        )
                        await self.call_counter.pop(oldest_key)

                    await self.clean_expired_cache_entries()

                    self.cache[key] = (
                        result,
                        datetime.utcnow() + timedelta(seconds=self.cache_lifetime),
                    )
                    await self._save_file_cache()

                    logging.debug(f"Result cached for key: {key}")
                    return result
            except Exception as e:
                logging.error(f"Error in cache logic for key {key}: {e}")
                raise

    async def clean_expired_cache_entries(self):
        """
        Cleans expired entries from both in-memory and file caches asynchronously.
        This method iterates over the cache entries and removes those that have expired.
        """
        try:
            current_time = datetime.utcnow()
            expired_keys = [
                key
                for key, (_, expiration_time) in self.cache.items()
                if expiration_time < current_time
            ]
            for key in expired_keys:
                del self.cache[key]
                logging.debug(f"Removed expired cache entry: {key}")

            expired_keys_file_cache = [
                key
                for key, (_, expiration_time) in self.file_cache.items()
                if expiration_time < current_time
            ]
            for key in expired_keys_file_cache:
                del self.file_cache[key]
            if expired_keys_file_cache:
                await self._save_file_cache()
            logging.debug(
                f"Expired entries removed from file cache: {expired_keys_file_cache}"
            )
        except Exception as e:
            logging.error(f"Error cleaning expired cache entries: {e}")

    def _evict_lru_from_memory(self):
        """
        Evicts the least recently used (LRU) item from the in-memory cache.
        This method is called when the cache exceeds its maximum size to maintain the cache size.
        """
        try:
            if len(self.cache) > self.cache_maxsize:
                lru_key = next(iter(self.cache))
                del self.cache[lru_key]
                logging.debug(f"LRU item evicted from in-memory cache: {lru_key}")
        except Exception as e:
            logging.error(f"Error evicting LRU item from memory: {e}")

    async def maintain_cache(self):
        """
        Periodically checks and maintains the cache by cleaning expired entries and evicting LRU items.
        This method runs in a separate asynchronous task to ensure efficient cache management.
        """
        try:
            # This enhanced block continuously monitors and maintains the cache, incorporating advanced logic to
            # determine the program's execution context and initiate a graceful shutdown if the cache remains unchanged
            # for an extended period, indicating potential program inactivity or improper closure.

            # Initialize variables to track cache changes and manage shutdown logic
            unchanged_cycles = (
                0  # Counter for the number of cycles with no cache changes
            )
            is_main_program = (
                __name__ == "__main__"
            )  # Check if running as the main program
            sleep_duration = (
                60 if is_main_program else 10
            )  # Sleep duration based on execution context

            # Continuously perform cache maintenance tasks with advanced monitoring for inactivity
            while True:
                await asyncio.sleep(
                    sleep_duration
                )  # Pause execution for the determined duration

                # Capture the state of caches before maintenance to detect changes
                initial_memory_cache_state = str(self.cache)
                initial_file_cache_state = str(self.file_cache)

                # Execute cache maintenance operations
                await self.clean_expired_cache_entries()  # Clean expired entries from caches
                self._evict_lru_from_memory()  # Evict least recently used items from memory cache

                # Determine if the cache states have changed following maintenance operations
                memory_cache_changed = initial_memory_cache_state != str(self.cache)
                file_cache_changed = initial_file_cache_state != str(self.file_cache)

                # Log the completion of cache maintenance based on the program's execution context
                context_message = (
                    "the main program"
                    if is_main_program
                    else "the imported module context"
                )
                logging.debug(f"Cache maintenance completed in {context_message}.")

                # Update the unchanged cycles counter based on cache state changes
                if memory_cache_changed or file_cache_changed:
                    unchanged_cycles = 0  # Reset counter if changes were detected
                else:
                    unchanged_cycles += (
                        1  # Increment counter if no changes were detected
                    )

                # Initiate a graceful shutdown if caches have remained unchanged for 10 cycles
                if unchanged_cycles >= 10:
                    logging.info(
                        "No cache changes detected for 10 cycles. Initiating graceful shutdown."
                    )
                    # Placeholder for graceful shutdown logic
                    # This should include tasks such as closing database connections, stopping async tasks,
                    # and any other cleanup required before safely terminating the program.
                    await self.initiate_graceful_shutdown()
                    break  # Exit the loop to stop further cache maintenance

        except Exception as e:
            # Log any exceptions encountered during the cache maintenance and monitoring process
            logging.error(f"Error in cache maintenance task: {e}")

    async def attempt_cache_retrieval(self, key: Tuple[Any, ...]) -> Optional[Any]:
        """
        Attempts to retrieve the cached result for the given key from both in-memory and file caches.
        This method checks the caches and returns the cached result if found.
        """
        try:
            if key in self.cache:
                logging.debug(f"In-memory cache hit for {key}")
                return self.cache[key]
            elif key in self.file_cache:
                logging.debug(f"File cache hit for {key}")
                result = self.file_cache[key]
                if len(self.cache) >= self.cache_maxsize:
                    self.cache.popitem(last=False)
                self.cache[key] = result
                return result
            return None
        except Exception as e:
            logging.error(f"Error attempting cache retrieval for key {key}: {e}")
            return None

    async def update_cache(
        self, key: Tuple[Any, ...], result: Any, is_async: bool = True
    ):
        """
        Updates the cache with the given key and result, managing cache sizes, evictions, and time-based expiration.
        This method handles both in-memory and file cache updates, including time-based eviction.
        """
        try:
            current_time = datetime.utcnow()
            expiration_time = current_time + timedelta(seconds=self.cache_lifetime)
            if len(self.cache) >= self.cache_maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = (result, expiration_time)
            if self.call_counter.get(key, 0) >= self.call_threshold:
                self.call_counter[key] = 0
                if is_async:
                    await self._save_to_file_cache(key, (result, expiration_time))
            else:
                self.background_cache_persistence(key, (result, expiration_time))
                logging.debug(
                    f"Cache updated for key: {key}. Cache size: {len(self.cache)}"
                )
        except Exception as e:
            logging.error(f"Error updating cache for key {key}: {e}")

    async def _save_to_file_cache(
        self, key: Tuple[Any, ...], value: Tuple[Any, datetime]
    ):
        """
        Asynchronously saves a key-value pair to the file cache.
        This method uses asynchronous file operations to ensure the event loop is not blocked.
        """
        try:
            self.file_cache[key] = value
            async with aiofiles.open(self.file_cache_path, "wb") as f:
                await asyncio.to_thread(
                    pickle.dump, self.file_cache, f, pickle.HIGHEST_PROTOCOL
                )
            logging.debug(f"File cache asynchronously updated for key: {key}")
        except Exception as e:
            logging.error(
                f"Error saving to file cache asynchronously for key {key}: {e}"
            )

    def _save_to_file_cache_sync(
        self, key: Tuple[Any, ...], value: Tuple[Any, datetime]
    ):
        """
        Synchronously saves a key-value pair to the file cache.
        This method is used for background cache persistence where asynchronous operations are not feasible.
        """
        try:
            self.file_cache[key] = value
            with open(self.file_cache_path, "wb") as f:
                pickle.dump(self.file_cache, f, pickle.HIGHEST_PROTOCOL)
            logging.debug(f"File cache synchronously updated for key: {key}")
        except Exception as e:
            logging.error(
                f"Error saving to file cache synchronously for key {key}: {e}"
            )

    def background_cache_persistence(self, key, value):
        """
        Initiates a background thread to persist cache updates to the file cache synchronously.
        This method is used when asynchronous operations are not feasible or preferred.
        """
        try:
            threading.Thread(
                target=self._save_to_file_cache_sync, args=(key, value)
            ).start()
        except Exception as e:
            logging.error(
                f"Error initiating background cache persistence for key {key}: {e}"
            )

    def dynamic_retry_strategy(self, exception: Exception) -> Tuple[int, int]:
        """
        Determines the retry strategy dynamically based on the exception type.

        This method is designed to adapt the retry strategy based on the type of exception encountered during
        the execution of the decorated function. It leverages the logging module to provide detailed insights
        into the decision-making process, ensuring that adjustments to the retry strategy are well-documented
        and traceable.

        Args:
            exception (Exception): The exception that triggered the retry logic.

        Returns:
            Tuple[int, int]: A tuple containing the number of retries and delay in seconds, representing
            the dynamically adjusted retry strategy based on the exception type.

        Detailed logging is performed to trace the decision-making process, providing insights into the
        adjustments made to the retry strategy based on the encountered exception type. This facilitates
        a deeper understanding of the retry mechanism's behavior in response to different failure scenarios.
        """

        # Log the initiation of the dynamic retry strategy determination process.
        logging.debug(
            f"Initiating dynamic retry strategy determination for exception: {exception}"
        )

        # Default retry strategy, applied when the exception type does not match any specific conditions.
        default_strategy = (self.retries, self.delay)
        logging.debug(
            f"Default retry strategy set to {default_strategy} retries and delay."
        )

        # Determine the retry strategy based on the exception type.
        if isinstance(exception, TimeoutError):
            # Adjust the retry strategy for TimeoutError exceptions.
            timeout_strategy = (
                5,
                1,
            )  # More retries with a short delay for timeout errors.
            logging.debug(
                f"TimeoutError encountered. Adjusting retry strategy to {timeout_strategy}."
            )
            return timeout_strategy

        elif isinstance(exception, ConnectionError):
            # Adjust the retry strategy for ConnectionError exceptions.
            connection_strategy = (
                3,
                5,
            )  # Fewer retries with a longer delay for connection errors.
            logging.debug(
                f"ConnectionError encountered. Adjusting retry strategy to {connection_strategy}."
            )
            return connection_strategy

        else:
            # Apply the default retry strategy for all other exception types.
            logging.debug(
                f"No specific adjustment made for exception type {type(exception)}. Applying default retry strategy."
            )
            return default_strategy

    async def log_performance(
        self, func: F, start_time: float, end_time: float
    ) -> None:
        """
        Asynchronously logs the performance of the decorated function, adjusting for decorator overhead.
        This method ensures thread safety and non-blocking I/O operations for logging performance metrics
        to a file. It dynamically calculates the overhead introduced by the decorator to provide accurate
        performance metrics.

        Args:
            func (F): The function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.

        Returns:
            None: This method does not return any value.
        """
        # Initialize logging with asynchronous capabilities to ensure non-blocking operations
        logging.debug("Asynchronous performance logging initiated")

        # Check if performance logging is enabled
        if self.enable_performance_logging:
            # Dynamically calculate the overhead introduced by the decorator
            # This overhead calculation could be refined based on extensive profiling
            overhead = 0.0001  # Example overhead value
            adjusted_time = end_time - start_time - overhead

            # Ensure thread safety with asynchronous file operations
            try:
                # Use aiofiles for asynchronous file I/O
                async with aiofiles.open("performance.log", "a") as f:
                    # Write the performance log asynchronously
                    await f.write(f"{func.__name__} executed in {adjusted_time:.6f}s\n")
                # Log the adjusted execution time for the function
                logging.debug(f"{func.__name__} executed in {adjusted_time:.6f}s")
            except Exception as e:
                # Log any exceptions encountered during the logging process
                logging.error(f"Error logging performance for {func.__name__}: {e}")
        else:
            # Log the decision not to log performance due to configuration
            logging.debug("Performance logging is disabled; skipping logging.")
        return None

    def __call__(self, func: F) -> F:
        """
        Makes the class instance callable, allowing it to be used as a decorator. It wraps the
        decorated function in a new function that performs argument validation, caching, logging,
        retry logic, and performance monitoring before executing the original function.

        This method dynamically adapts to both synchronous and asynchronous functions, ensuring
        that all enhanced functionalities are applied consistently across different types of function
        executions. It leverages internal methods for argument validation, caching logic, retry mechanisms,
        and performance logging to provide a comprehensive enhancement to the decorated function.

        Args:
            func (F): The function to be decorated.

        Returns:
            F: The wrapped function, which includes enhanced functionality.
        """

        async def run_async_coroutine(coroutine):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    return asyncio.ensure_future(coroutine, loop=loop)
            except RuntimeError:
                return asyncio.run(coroutine)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                logging.info(
                    f"Async call to {func.__name__} with args: {args} and kwargs: {kwargs}"
                )
                if self.enable_validation:
                    await self._validate_func_signature(func, *args, **kwargs)
                try:
                    start_time = time.perf_counter()
                    result = await self.wrapper_logic(func, True, *args, **kwargs)
                    end_time = time.perf_counter()
                    await self.log_performance(func, start_time, end_time)
                    return result
                except Exception as e:
                    logging.error(f"Exception in async call to {func.__name__}: {e}")
                    raise
                finally:
                    self.end_time = time.perf_counter()

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                logging.info(
                    f"Sync call to {func.__name__} with args: {args} and kwargs: {kwargs}"
                )
                if self.enable_validation:
                    asyncio.run_coroutine_threadsafe(
                        self._validate_func_signature(func, *args, **kwargs),
                        asyncio.get_event_loop(),
                    )
                try:
                    start_time = time.perf_counter()
                    result = asyncio.run_coroutine_threadsafe(
                        self.wrapper_logic(func, False, *args, **kwargs),
                        asyncio.get_event_loop(),
                    ).result()
                    end_time = time.perf_counter()
                    asyncio.run_coroutine_threadsafe(
                        self.log_performance(func, start_time, end_time),
                        asyncio.get_event_loop(),
                    )
                    return result
                except Exception as e:
                    logging.error(f"Exception in sync call to {func.__name__}: {e}")
                    raise

            return sync_wrapper

    async def wrapper_logic(self, func: F, is_async: bool, *args, **kwargs) -> Any:
        """
        Contains the core logic for retrying, caching, logging, validating, and monitoring the execution
        of the decorated function. This method dynamically adapts to both synchronous and asynchronous functions,
        ensuring that the execution logic is seamlessly applied regardless of the function's nature.

        The method's functionality includes:
        - Argument validation to ensure compliance with specified criteria.
        - Caching logic for efficient retrieval of function results, minimizing execution time for repeated calls.
        - Retry mechanisms to address transient failures by re-executing the function according to predefined rules.
        - Performance logging for insights into execution efficiency.

        Args:
            func (F): The function to be executed.
            is_async (bool): Indicates if the function is asynchronous.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function execution, either from cache or newly computed.
        """
        # Initialize performance monitoring
        start_time = time.perf_counter()
        cache_key = self.generate_cache_key(
            func, args, kwargs
        )  # Assuming generate_cache_key method exists

        # Ensure thread safety with asyncio.Lock for cache operations
        async with self.cache_lock:
            # Cache logic initialization
            if self.enable_caching:
                cache_key = self.cache_key_strategy(func, args, kwargs)
                cached_result = await self.attempt_cache_retrieval(cache_key)
                if cached_result is not None:
                    logging.info(f"Cache hit for {func.__name__} with key {cache_key}")
                    return cached_result
                else:
                    logging.info(f"Cache miss for {func.__name__} with key {cache_key}")

            # Retry Logic
            if self.dynamic_retry_enabled:
                retries, delay = self.dynamic_retry_strategy(Exception)

            # Initialize retry attempt counter
            attempt = 0
            while attempt <= self.retries:
                try:
                    if is_async:
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    # Cache the result if caching is enabled
                    if self.enable_caching:
                        await self.update_cache(cache_key, result, is_async)
                        logging.info(
                            f"Result cached for {func.__name__} with key {cache_key}"
                        )

                    return result
                except self.retry_exceptions as e:
                    logging.warning(
                        f"Retry {attempt + 1} for {func.__name__} due to {e}"
                    )
                    if attempt < self.retries:
                        if is_async:
                            await asyncio.sleep(delay)
                        else:
                            time.sleep(delay)
                    attempt += 1
                except Exception as e:
                    logging.error(f"Exception during {func.__name__} execution: {e}")
                    raise e
                finally:
                    # Performance monitoring
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    logging.info(
                        f"Execution of {func.__name__} completed in {execution_time:.2f}s"
                    )
                    if is_async:
                        await self.log_performance(func, start_time, end_time)
                    else:
                        # Synchronous logging of performance
                        self.log_performance_sync(func, start_time, end_time)

    async def _validate_func_signature(self, func, *args, **kwargs):
        """
        Validates the function's signature against provided arguments and types.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions.
        It leverages Python's introspection capabilities to validate function signatures dynamically.

        Args:
            func (Callable): The function whose signature is being validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If the provided arguments do not match the function's signature.

        Returns:
            None: This method does not return any value but raises an exception on failure.
        """
        # Capture the start time for performance logging
        start_time = time.perf_counter()
        logging.debug(
            f"Validating function signature for {func.__name__} at {start_time}"
        )

        # Retrieve the function's signature and bind the provided arguments
        sig = signature(func)
        logging.debug(f"Function signature: {sig}")
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        logging.debug(f"Bound arguments with defaults applied: {bound_args}")

        # Retrieve type hints and validate each argument against its expected type
        type_hints = get_type_hints(func)
        logging.debug(f"Type hints: {type_hints}")
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name, None)  # Set a default value of None
            logging.debug(
                f"Validating argument '{name}' with value '{value}' against expected type '{expected_type}'"
            )
            if expected_type is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
                )

        # Log the completion of the validation process
        end_time = time.perf_counter()
        logging.debug(
            f"Validation of function signature for {func.__name__} completed in {end_time - start_time:.2f}s"
        )

    async def _get_arg_position(self, func: F, arg_name: str) -> int:
        """
        Determines the position of an argument in the function's signature.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions.
        It leverages Python's introspection capabilities to dynamically determine argument positions.

        Args:
            func (Callable): The function being inspected.
            arg_name (str): The name of the argument whose position is sought.

        Returns:
            int: The position of the argument in the function's signature.

        Raises:
            ValueError: If the argument name is not found in the function's signature.
        """
        # Capture the start time for performance logging
        start_time = time.perf_counter()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} at {start_time}"
        )

        # Determine the position of the argument in the function's signature
        parameters = list(signature(func).parameters)
        if arg_name not in parameters:
            raise ValueError(
                f"Argument '{arg_name}' not found in {func.__name__}'s signature"
            )
        result = parameters.index(arg_name)

        # Log the determined position
        logging.debug(f"Argument position for {arg_name} in {func.__name__}: {result}")

        # Log the completion of the process
        end_time = time.perf_counter()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} completed in {end_time - start_time:.2f}s"
        )
        return result

    async def validate_arguments(self, func: F, *args, **kwargs) -> None:
        """
        Validates the arguments passed to a function against expected type hints and custom validation rules.
        Adjusts for whether the function is a bound method (instance or class method) or a regular function
        or static method, and applies argument validation accordingly.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions,
        leveraging asyncio for non-blocking operations and ensuring thread safety with asyncio.Lock.

        Args:
            func (Callable): The function whose arguments are to be validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If an argument does not match its expected type.
            ValueError: If an argument fails custom validation rules.
        """
        # Initialize performance logging
        start_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validating arguments for {func.__name__} at {start_time} with args: {args} and kwargs: {kwargs}",
            extra={"async_mode": True},
        )

        # Adjust args for bound methods (instance or class methods)
        if inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        ):
            # For bound methods, the first argument ('self' or 'cls') should not be included in validation
            args = args[1:]

        # Attempt to bind args and kwargs to the function's signature
        try:
            bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
        except TypeError as e:
            logging.error(
                f"Error binding arguments for {func.__name__}: {e}",
                extra={"async_mode": True},
            )
            raise

        bound_args.apply_defaults()
        type_hints = get_type_hints(func)

        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name)
            if not await self.validate_type(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type '{expected_type}', got type '{type(value)}'"
                )

            validation_rule = self.validation_rules.get(name)
            if validation_rule:
                valid = (
                    await validation_rule(value)
                    if asyncio.iscoroutinefunction(validation_rule)
                    else validation_rule(value)
                )
                if not valid:
                    raise ValueError(
                        f"Validation failed for argument '{name}' with value '{value}'"
                    )

        end_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validation completed at {end_time} taking total time of {end_time - start_time} seconds",
            extra={"async_mode": True},
        )

    async def validate_type(self, value: Any, expected_type: Any) -> bool:
        """
        Recursively validates a value against an expected type, handling generics, special forms, and complex types.
        This method is meticulously designed to be exhaustive in its approach to type validation,
        ensuring compatibility with a wide range of type annotations, including generics, special forms, and complex types.
        It leverages Python's typing module to interpret and validate against the provided type hints accurately.
        Utilizes asyncio for non-blocking operations and ensures thread safety with asyncio.Lock.

        Args:
            value (Any): The value to validate.
            expected_type (Any): The expected type against which to validate the value.

        Returns:
            bool: True if the value matches the expected type, False otherwise.
        """
        start_time = asyncio.get_event_loop().time()
        # Early exit for typing.Any, indicating any type is acceptable.
        if expected_type is Any:
            logging.debug(
                "Any type encountered, validation passed.", extra={"async_mode": True}
            )
            return True

        # Handle Union types, including Optional, by validating against each type argument until one matches.
        if get_origin(expected_type) is Union:
            logging.debug(
                f"Union type encountered: {expected_type}", extra={"async_mode": True}
            )
            return any(
                await self.validate_type(value, arg) for arg in get_args(expected_type)
            )

        # Handle special forms like Any, ClassVar, etc., assuming validation passes for these.
        if isinstance(expected_type, _SpecialForm):
            logging.debug(
                f"Special form encountered: {expected_type}", extra={"async_mode": True}
            )
            return True

        # Extract the origin type and type arguments from the expected type, if applicable.
        origin_type = get_origin(expected_type)
        type_args = get_args(expected_type)

        # Handle generic types (List[int], Dict[str, Any], etc.)
        if origin_type is not None:
            if not isinstance(value, origin_type):
                logging.debug(
                    f"Value {value} does not match the origin type {origin_type}.",
                    extra={"async_mode": True},
                )
                return False
            if type_args:
                # Validate type arguments (e.g., the 'int' in List[int])
                if issubclass(origin_type, collections.abc.Mapping):
                    key_type, val_type = type_args
                    logging.debug(
                        f"Validating Mapping with key type {key_type} and value type {val_type}.",
                        extra={"async_mode": True},
                    )
                    return all(
                        await self.validate_type(k, key_type)
                        and await self.validate_type(v, val_type)
                        for k, v in value.items()
                    )
                elif issubclass(
                    origin_type, collections.abc.Iterable
                ) and not issubclass(origin_type, (str, bytes, bytearray)):
                    element_type = type_args[0]
                    logging.debug(
                        f"Validating each element in Iterable against type {element_type}.",
                        extra={"async_mode": True},
                    )
                    return all(
                        [await self.validate_type(elem, element_type) for elem in value]
                    )
                # Extend to handle other generic types as needed
        else:
            # Handle non-generic types directly
            if not isinstance(value, expected_type):
                logging.debug(
                    f"Value {value} does not match the expected non-generic type {expected_type}.",
                    extra={"async_mode": True},
                )
                return False
            return True

        # Fallback for unsupported types
        logging.debug(
            f"Type {expected_type} not supported by the current validation logic.",
            extra={"async_mode": True},
        )
        return False


@StandardDecorator()
def sync_example(input_value: int, recursion_count: int) -> List[int]:
    """
    Performs a series of complex nested recursive calculations based on the input value.
    Utilizes a variety of mathematical operations to simulate complex logic.

    Args:
        input_value (int): The initial value for the calculation.
        recursion_count (int): The depth of recursion to perform.

    Returns:
        List[int]: A list of calculated values at each recursion step.
    """

    def complex_calculate(
        value: int, depth: int, operator_list: List[str]
    ) -> List[Tuple[int, int, str]]:
        """
        Inner function to perform recursive calculations based on a set of operators.

        Args:
            value (int): The current value to be operated on.
            depth (int): The current depth of recursion.
            operator_list (List[str]): A list of operators to apply to the value.

        Returns:
            List[Tuple[int, int, str]]: A list of tuples containing the recursion depth, result, and operator used.
        """
        # Initialize the result list to store tuples of (recursion depth, calculation result, operator used)
        result = []

        # Base case: if the recursion depth reaches 0, return the current value with an empty operator string
        if depth == 0:
            return [(depth, value, "")]

        # Iterate through each operator in the operator list to perform calculations
        for operator in operator_list:
            try:
                if operator == "+":
                    result.append((value, value + 1, operator))
                    complex_calculate(value + 1, depth - 1, operator_list)
                elif operator == "-":
                    result.append((value, value - 1, operator))
                    complex_calculate(value - 1, depth - 1, operator_list)
                elif operator == "*":
                    result.append((value, value * 2, operator))
                    complex_calculate(value * 2, depth - 1, operator_list)
                elif operator == "/":
                    result.append((value, value / 2, operator))
                    complex_calculate(value / 2, depth - 1, operator_list)
                elif operator == "**":
                    result.append((value, value**2, operator))
                    complex_calculate(value**2, depth - 1, operator_list)
                elif operator == "//":
                    result.append((value, value // 2, operator))
                    complex_calculate(value // 2, depth - 1, operator_list)
                elif operator == "%":
                    result.append((value, value % 2, operator))
                    complex_calculate(value % 2, depth - 1, operator_list)
                elif operator == "^":
                    result.append((value, value ^ 2, operator))
                    complex_calculate(value ^ 2, depth - 1, operator_list)
                elif operator == "&":
                    result.append((value, value & 2, operator))
                    complex_calculate(value & 2, depth - 1, operator_list)
                elif operator == "|":
                    result.append((value, value | 2, operator))
                    complex_calculate(value | 2, depth - 1, operator_list)
                elif operator == "<<":
                    result.append((value, value << 2, operator))
                    complex_calculate(value << 2, depth - 1, operator_list)
                elif operator == ">>":
                    result.append((value, value >> 2, operator))
                    complex_calculate(value >> 2, depth - 1, operator_list)
                # The '>>>' operator is not recognized in Python as it does not support unsigned right shift directly.
                # To simulate '>>>' behavior, we first ensure the value is treated as unsigned by applying a mask,
                # then perform a standard right shift.
                elif operator == ">>>":
                    # Applying a mask to ensure the value is treated as unsigned.
                    unsigned_value = value & 0xFFFFFFFF
                    # Performing a standard right shift on the masked value.
                    shifted_value = unsigned_value >> 2
                    result.append(
                        (value, shifted_value, operator)
                    )  # Simulated unsigned right shift
                    # Passing the shifted value to the recursive call.
                    complex_calculate(shifted_value, depth - 1, operator_list)
                elif operator == "~":
                    result.append((value, ~value, operator))
                    complex_calculate(~value, depth - 1, operator_list)
                elif operator == "not":
                    result.append((value, not value, operator))
                    complex_calculate(not value, depth - 1, operator_list)
                elif operator == "and":
                    result.append((value, value and 1, operator))
                    complex_calculate(value and 1, depth - 1, operator_list)
                elif operator == "or":
                    result.append((value, value or 1, operator))
                    complex_calculate(value or 1, depth - 1, operator_list)
                elif operator == "if":
                    result.append((value, value if value else 1, operator))
                    complex_calculate(value if value else 1, depth - 1, operator_list)
                elif operator == "else":
                    result.append((value, 1 if value else value, operator))
                    complex_calculate(1 if value else value, depth - 1, operator_list)
                elif operator == "elif":
                    result.append((value, 1 if value else value, operator))
                    complex_calculate(1 if value else value, depth - 1, operator_list)
                elif operator == "while":
                    result.append((value, value, operator))
                    complex_calculate(value, depth - 1, operator_list)
                elif operator == "for":
                    result.append((value, value, operator))
                    complex_calculate(value, depth - 1, operator_list)

            except Exception as e:
                # In case of any exception, append a tuple with the current depth, 0 as result, and the exception message
                result.append((depth, 0, str(e)))
                continue  # Continue with the next operator

        # Return the result list containing tuples of (recursion depth, calculation result, operator used)
        return result

    # Define the initial list of operators to be used in the calculations
    operator_list = [
        "+",
        "-",
        "*",
        "/",
        "**",
        "//",
        "%",
        "^",
        "&",
        "|",
        "<<",
        ">>",
        ">>>",
        "~",
        "not",
        "and",
        "or",
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
        "is",
        "is not",
        "in",
        "not in",
        "and",
        "or",
        "if",
        "else",
        "elif",
        "while",
        "for",
        "with",
        "as",
    ]

    # Call the complex_calculate function with the initial input value, recursion count, and operator list
    # Return the result of the calculations
    return complex_calculate(input_value, recursion_count, operator_list)


# Test 2: Asynchronous function with result caching
@StandardDecorator()
async def async_example(input_value: int, iterations: int) -> List[int]:
    """
    Asynchronously performs calculations based on the input value, with results cached for unique inputs.
    Simulates I/O operations to demonstrate asynchronous execution and caching.

    Args:
        input_value (int): The input value for the calculation.
        iterations (int): The number of iterations to perform.

    Returns:
        List[int]: The result of the calculation for each iteration.
    """
    result = []
    for _ in range(iterations):
        await asyncio.sleep(0.1)  # Simulate an I/O operation
        calculated_value = input_value**2  # Example calculation
        result.append(calculated_value)

    return result


# Test Runner
@StandardDecorator()
async def test_standard_decorator_async():
    """
    Asynchronous wrapper for testing both synchronous and asynchronous decorated functions.
    """
    # Synchronous function test
    try:
        sync_result = sync_example(5, 3)
        print(f"Sync Example Result: {sync_result}")
    except Exception as e:
        print(f"Sync Example Failed: {e}")

    # Asynchronous function test
    try:
        async_result = await async_example(4, 5)
        print(f"Async Example Result: {async_result}")
    except Exception as e:
        print(f"Async Example Failed: {e}")


async def main():
    await test_standard_decorator_async()


if __name__ == "__main__":
    # Initialize resource profiling
    tracemalloc.start()

    # Simplified event loop management and graceful shutdown
    asyncio.run(main())

    logging.info("Program terminated")

"""
================================================================================
Module Footer Documentation for StandardDecorator
================================================================================

    TODO:
    ================================================================================
    High Priority:
        Security:
            - [ ] Ensure that logging does not inadvertently expose sensitive information.
        Documentation:
            - [ ]
        Optimization:
            - [ ] 
        Flexibility:
            - [ ] Implement a mechanism to allow users to define custom cache eviction policies.
        Automation:
            - [ ] Automate the testing of the decorator with different configurations.
        Scalability:
            - [ ] Evaluate the decorator's performance in high-load scenarios.
        Ethics:
            - [ ] Review the logging and caching mechanisms for potential ethical concerns.
        Bug Fix:
            - [ ] Address any reported bugs related to caching and retry mechanisms.
        Robustness:
            - [ ] 
        Clean Code:
            - [ ] 
        Stability:
            - [ ] Conduct stress tests to ensure the decorator's stability under various conditions.
        Formatting:
            - [ ] 
        Logics:
            - [ ] Review the logic for potential logical flaws or inefficiencies.
        Integration:
            - [ ] Test integration with other components of the project.

    Medium Priority:
        Performance:
            - [ ] 
        Usability:
            - [ ] Develop a user-friendly interface for configuring the decorator parameters.
            - [ ] Provide detailed examples of advanced use cases in the documentation.
            - [ ] Develop a GUI tool for configuring and testing the decorator.
            - [ ] Implement a mechanism to track and visualize the decorator's performance metrics.
            - [ ] Implement automatic total reporting mechanisms and data visualisations.
        Testing:
            - [ ] Increase unit test coverage to include all decorator functionalities.
        Compliance:
            - [ ] 
        Accessibility:
            - [ ] Improve documentation accessibility and readability.
            - [ ] Implement accessibility features for users with disabilities.
        Internationalization:
            - [ ] Add support for internationalization in logging messages.
            - [ ] Translate the documentation into multiple languages.

    Low Priority:
        Extensibility:
            - [ ] Explore mechanisms to allow third-party extensions of the decorator functionalities.
            - [ ] Implement a plugin system for adding custom validation rules and retry strategies.
        Community:
            - [ ] Establish a feedback loop with users to gather insights on potential improvements.
            - [ ] Create a community forum for users to share experiences and best practices.
        Documentation:
            - [ ] Create a comprehensive FAQ section addressing common issues and questions.
            - [ ] Add a glossary of terms and concepts used in the decorator documentation.
            - [ ] Include real-world use cases and success stories in the documentation.
        Optimization:
            - [ ] Research advanced Python optimization techniques for potential application.
            - [ ] Investigate caching libraries and algorithms for performance improvements.
            - [ ] Optimize the decorator for memory usage and efficiency.
            - [ ] Implement lazy loading of resources to improve performance.
            - [ ] Evaluate the decorator's performance in low-resource environments.
            - [ ] Optimize the decorator for multi-threaded and multi-process environments.
            - [ ] Investigate the use of caching proxies for distributed caching.
            - [ ] Implement a mechanism to track and visualize the decorator's resource usage.

    Routine:
        Code Reviews:
            - [ ] Conduct regular code reviews to maintain code quality and consistency.
            - [ ] Review the decorator for adherence to coding standards and best practices.
            - [ ] Collaborate with team members to identify and address code quality issues.
        Dependency Updates:
            - [ ] Regularly update dependencies to their latest stable versions.
            - [ ] Review changelogs and release notes for potential breaking changes.
            - [ ] Test the decorator with updated dependencies to ensure compatibility.
            - [ ] Monitor security advisories for dependencies and apply patches promptly.
            - [ ] Automate dependency updates and testing to streamline the process.
            - [ ] Implement a mechanism to track and visualize the decorator's dependency versions.
            - [ ] Evaluate the impact of dependency updates on the decorator's performance and stability.
        Documentation Updates:
            - [ ] Keep the documentation up to date with the latest changes and additions.
            - [ ] Review the documentation for accuracy, clarity, and completeness.
            - [ ] Add examples and use cases to illustrate the decorator's functionalities.
            - [ ] Include troubleshooting tips and solutions for common issues.
        Community Engagement:
            - [ ] Engage with the community through forums, GitHub issues, and social media.
            - [ ] Respond to user feedback and feature requests in a timely manner.
            - [ ] Encourage contributions from the community through documentation and code improvements.
            - [ ] Organize webinars, workshops, and hackathons to promote the decorator.
        Security Audits:
            - [ ] Perform periodic security audits to identify and mitigate potential vulnerabilities.
            - [ ] Conduct penetration testing and code reviews to ensure robust security measures.
            - [ ] Implement security best practices and guidelines in the decorator.
            - [ ] Monitor security advisories and alerts for potential threats and risks.

    Known Issues:
        Security:
            - [ ] Review for potential security vulnerabilities in the caching mechanism.
        Documentation:
            - [ ] Update the documentation to reflect the latest changes and features.
        Optimization:
            - [ ] Identify areas for performance improvement.
        Flexibility:
            - [ ] Assess the need for additional configuration options.
        Automation:
            - [ ] Improve the automation of test case execution.
        Scalability:
            - [ ] Investigate scalability issues reported by users.
        Ethics:
            - [ ] Ensure compliance with data protection regulations.
        Bug Fix:
            - [ ] Fix known bugs listed in the issue tracker.
        Robustness:
            - [ ] Address issues related to the robustness of the retry mechanism.
        Clean Code:
            - [ ] Continuously refactor the codebase for cleanliness.
        Stability:
            - [ ] Resolve issues that cause instability in specific scenarios.
        Formatting:
            - [ ] Ensure all code conforms to the project's formatting standards.
        Logics:
            - [ ] Validate the logical flow and correctness of the implementation.
        Integration:
            - [ ] Address integration challenges with other project modules.
    ================================================================================
"""


class AsyncFileHandler(logging.Handler):
    """
    An asynchronous logging handler using aiofiles for non-blocking file writing.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initializes the AsyncFileHandler.

        Args:
            filename (str): The name of the file to which logs are written.
            mode (str): The mode in which the file is opened.
            loop (Optional[asyncio.AbstractEventLoop]): The asyncio event loop. If not provided, the current running loop is used.
        """
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.loop = loop or asyncio.get_event_loop()
        self.aiofile = None  # Initialization moved to aio_open due to async nature

    async def aio_open(self):
        """
        Asynchronously opens the file with aiofiles.
        """
        self.aiofile = await aiofiles.open(self.filename, self.mode)

    async def aio_write(self, msg: str):
        """
        Asynchronously writes a log message to the file.

        Args:
            msg (str): The log message to write.
        """
        if self.aiofile is None:
            await self.aio_open()
        await self.aiofile.write(msg)
        await self.aiofile.flush()

    def emit(self, record):
        """
        Overrides the emit method to write logs asynchronously.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        msg = self.format(record)
        asyncio.run_coroutine_threadsafe(self.aio_write(msg + "\n"), self.loop)


class ResourceProfiler:
    """
    Profiles resource usage (CPU, memory, etc.) and logs the metrics at specified intervals.
    """

    def __init__(self, interval: int = 60, output: str = "resource_usage.log"):
        """
        Initializes the resource profiler.

        Args:
            interval (int): The interval (in seconds) at which to log resource usage.
            output (str): The file path for resource profiling logs.
        """
        self.interval = interval
        self.output = output
        self.running = False
        self.lock = asyncio.Lock()  # Ensures thread safety in async context

    async def start(self):
        """
        Starts the resource profiling in a background coroutine.
        """
        async with self.lock:
            if not self.running:
                self.running = True
                tracemalloc.start()
                asyncio.create_task(self._profile())

    async def stop(self):
        """
        Stops the resource profiling.
        """
        async with self.lock:
            if self.running:
                self.running = False
                tracemalloc.stop()

    async def _profile(self):
        """
        Periodically logs resource usage until stopped.
        """
        while self.running:
            await self._log_resource_usage()
            await asyncio.sleep(self.interval)

    async def _log_resource_usage(self):
        """
        Logs the current resource usage asynchronously.
        """
        process = psutil.Process()
        cpu_usage = psutil.cpu_percent()
        memory_usage = process.memory_percent()
        current, peak = tracemalloc.get_traced_memory()
        async with aiofiles.open(self.output, "a") as af:
            await af.write(
                f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Traced Memory: Current {current} bytes, Peak {peak} bytes\n"
            )
        logging.info(
            f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Traced Memory: Current {current} bytes, Peak {peak} bytes"
        )


class StandardDecorator:
    def __init__(
        self,
        retries: int = 3,
        delay: int = 1,
        cache_results: bool = True,
        log_level: int = logging.DEBUG,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        cache_maxsize: int = 100,
        enable_performance_logging: bool = True,
        enable_resource_profiling: bool = True,
        dynamic_retry_enabled: bool = True,
        cache_key_strategy: Optional[
            Callable[[Callable, Tuple[Any, ...], Dict[str, Any]], Tuple[Any, ...]]
        ] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
        file_cache_path: str = "file_cache.pkl",
        indefinite_operation: bool = False,
        resource_profiling_interval: int = 60,
        resource_profiling_output: str = "resource_usage.log",
    ):
        """
        Initializes the StandardDecorator with a wide range of configurations for retries, caching, logging, and more.

        Args:
            retries (int): The number of retries for the decorated function upon failure.
            delay (int): The delay between retries.
            cache_results (bool): Enables or disables caching of function results.
            log_level (int): The logging level.
            validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Custom validation rules for function arguments.
            retry_exceptions (Tuple[Type[BaseException], ...]): Exceptions that trigger a retry.
            cache_maxsize (int): The maximum size of the in-memory cache.
            enable_performance_logging (bool): Enables or disables performance logging.
            dynamic_retry_enabled (bool): Enables or disables dynamic retry strategies based on exception types.
            cache_key_strategy (Optional[Callable]): Custom strategy for generating cache keys.
            enable_caching (bool): Enables or disables caching functionality.
            enable_validation (bool): Enables or disables argument validation.
            file_cache_path (str): The file path for persistent cache storage.
            indefinite_operation (bool): Keeps the decorator active indefinitely for long-running operations.
            enable_resource_profiling (bool): Enables or disables resource profiling.
            resource_profiling_interval (int): The interval (in seconds) at which to log resource usage.
            resource_profiling_output (str): The file path for resource profiling logs.
        Raises:
            ValueError: If the provided configurations are invalid.
        """
        # Validate input parameters for sanity checks
        if retries < 0 or delay < 0:
            raise ValueError("Retries and delay must be non-negative.")

        # Initialize logging
        self.logger = self.setup_logging(level=log_level)

        # Signal handling for graceful shutdown
        if not indefinite_operation:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

        # Initialize caching mechanisms
        self.cache = {} if cache_results else None
        self.file_cache_path = file_cache_path
        self.cache_lock = (
            asyncio.Lock()
        )  # Use asyncio.Lock for thread safety in async context
        if cache_results and not os.path.exists(file_cache_path):
            with open(file_cache_path, "wb") as f:
                pickle.dump({}, f)

        # Other initializations
        self.retries = retries
        self.delay = delay
        self.validation_rules = validation_rules or {}
        self.retry_exceptions = retry_exceptions
        self.cache_maxsize = cache_maxsize
        self.enable_performance_logging = enable_performance_logging
        self.enable_resource_profiling = enable_resource_profiling
        self.dynamic_retry_enabled = dynamic_retry_enabled
        self.cache_key_strategy = cache_key_strategy or self._default_cache_key_strategy
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation
        self.indefinite_operation = indefinite_operation
        self.resource_profiling_interval = resource_profiling_interval
        self.resource_profiling_output = resource_profiling_output
        self._initialize_resource_profiling()
        self._initialize_cache()
        self._run_async_coroutine(self._initialize_async_components())
        self.logger.info("StandardDecorator fully initialized.")

    def _run_async_coroutine(self, coroutine):
        """
        Adapts the execution of a coroutine based on the current event loop state, ensuring compatibility with both
        synchronous and asynchronous contexts.

        Args:
            coroutine: The coroutine to be executed.

        Returns:
            The result of the coroutine execution if the event loop is not running; otherwise, schedules the coroutine
            for future execution.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return asyncio.ensure_future(coroutine, loop=loop)
        except RuntimeError:
            return asyncio.run(coroutine)

    def _signal_handler(self, signum, frame):
        """
        Signal handler for initiating graceful shutdown. This method is designed to be compatible with asynchronous
        operations and ensures that the shutdown process is handled properly in an asyncio context.

        Args:
            signum: The signal number received.
            frame: The current stack frame.
        """
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown.")
        asyncio.create_task(self._graceful_shutdown())

    async def _graceful_shutdown(self):
        """
        Handles graceful shutdown on receiving termination signals. This method logs the received signal and initiates
        a graceful shutdown process. It saves the cache to a file if caching is enabled and performs additional cleanup
        actions. Finally, it cancels all outstanding tasks and stops the asyncio event loop, ensuring a clean shutdown.
        """
        self.logger.info("Initiating graceful shutdown process.")
        # Perform necessary cleanup actions here
        # Save cache to file if caching is enabled
        if self.enable_caching and self.cache is not None:
            async with aiofiles.open(self.file_cache_path, "wb") as f:
                await f.write(
                    pickle.dumps(self.cache)
                )  # Corrected to use async write operation
            self.logger.info("Cache saved to file successfully.")
        # Additional cleanup actions can be added here

        # Cancel all outstanding tasks and stop the event loop
        await self._cancel_outstanding_tasks()

    async def _cancel_outstanding_tasks(self):
        """
        Cancels all outstanding tasks in the current event loop and stops the loop. This method retrieves all tasks in
        the current event loop, cancels them, and then gathers them to ensure they are properly handled. It logs the
        cancellation of tasks and the successful shutdown of the service.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            task
            for task in asyncio.all_tasks(loop)
            if task is not asyncio.current_task()
        ]

        for task in tasks:
            task.cancel()

        self.logger.info("Cancelling outstanding tasks.")
        await asyncio.gather(*tasks, return_exceptions=True)
        self.logger.info("Successfully shutdown service.")
        loop.stop()

    async def initialize_performance_logger(self):
        """
        Initializes the performance logger. This method is designed to prepare the logging environment
        for capturing and recording performance metrics asynchronously. It ensures that the necessary
        setup for performance logging is completed before any performance metrics are logged.

        This initialization includes setting up an asynchronous file handler for non-blocking I/O operations,
        ensuring that performance metrics can be logged without impacting the execution flow of the decorated functions.
        """
        # Setup an asynchronous file handler for performance logging
        self.performance_log_handler = AsyncFileHandler("performance.log", "a")
        await self.performance_log_handler.aio_open()
        logging.getLogger().addHandler(self.performance_log_handler)
        logging.debug("Performance logger initialized with asynchronous file handling.")

    async def log_performance(
        self, func: Callable, start_time: float, end_time: float
    ) -> None:
        """
        Asynchronously logs the performance of the decorated function, adjusting for decorator overhead.
        This method ensures thread safety and non-blocking I/O operations for logging performance metrics
        to a file. It dynamically calculates the overhead introduced by the decorator to provide accurate
        performance metrics.

        The logging operation is performed using an asynchronous file handler, ensuring that the logging
        process does not block the execution of the program. This method leverages the AsyncFileHandler
        class for asynchronous file operations, providing a non-blocking and thread-safe way to log
        performance metrics.

        Args:
            func (Callable): The function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.

        Returns:
            None: This method does not return any value.
        """
        # Initialize logging with asynchronous capabilities to ensure non-blocking operations
        logging.debug("Asynchronous performance logging initiated")

        # Check if performance logging is enabled
        if self.enable_performance_logging:
            # Dynamically calculate the overhead introduced by the decorator
            # This overhead calculation could be refined based on extensive profiling
            overhead = 0.0001  # Example overhead value
            adjusted_time = end_time - start_time - overhead

            # Construct the log message
            log_message = f"{func.__name__} executed in {adjusted_time:.6f}s\n"

            # Ensure thread safety with asynchronous file operations
            try:
                # Write the performance log asynchronously using the AsyncFileHandler
                await self.performance_log_handler.aio_write(log_message)
                # Log the adjusted execution time for the function
                logging.debug(f"{func.__name__} executed in {adjusted_time:.6f}s")
            except Exception as e:
                # Log any exceptions encountered during the logging process
                logging.error(f"Error logging performance for {func.__name__}: {e}")
        else:
            # Log the decision not to log performance due to configuration
            logging.debug("Performance logging is disabled; skipping logging.")
        return None

    def dynamic_retry_strategy(self, exception: Exception) -> Tuple[int, int]:
        """
        Determines the retry strategy dynamically based on the exception type.

        This method is designed to adapt the retry strategy based on the type of exception encountered during
        the execution of the decorated function. It leverages the logging module to provide detailed insights
        into the decision-making process, ensuring that adjustments to the retry strategy are well-documented
        and traceable.

        Args:
            exception (Exception): The exception that triggered the retry logic.

        Returns:
            Tuple[int, int]: A tuple containing the number of retries and delay in seconds, representing
            the dynamically adjusted retry strategy based on the exception type.

        Detailed logging is performed to trace the decision-making process, providing insights into the
        adjustments made to the retry strategy based on the encountered exception type. This facilitates
        a deeper understanding of the retry mechanism's behavior in response to different failure scenarios.
        """

        # Log the initiation of the dynamic retry strategy determination process.
        logging.debug(
            f"Initiating dynamic retry strategy determination for exception: {exception}"
        )

        # Default retry strategy, applied when the exception type does not match any specific conditions.
        default_strategy = (self.retries, self.delay)
        logging.debug(
            f"Default retry strategy set to {default_strategy} retries and delay."
        )

        # Determine the retry strategy based on the exception type.
        if isinstance(exception, TimeoutError):
            # Specific retry strategy for TimeoutError
            strategy = (5, 2)  # Example: 5 retries with 2 seconds delay
            logging.debug(
                f"TimeoutError encountered. Adjusting retry strategy to {strategy}."
            )
        elif isinstance(exception, ConnectionError):
            # Specific retry strategy for ConnectionError
            strategy = (3, 5)  # Example: 3 retries with 5 seconds delay
            logging.debug(
                f"ConnectionError encountered. Adjusting retry strategy to {strategy}."
            )
        else:
            # Fallback to default strategy for other exceptions
            strategy = default_strategy
            logging.debug(
                f"No specific strategy for {type(exception)}. Using default strategy {strategy}."
            )

        return strategy

    async def setup_logging(self):
        """
        Sets up asynchronous logging handlers, including terminal, asynchronous file, and rotating file handlers.
        This method ensures that logging does not block the execution of the program and is thread-safe.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Terminal handler setup for immediate console output
        terminal_handler = logging.StreamHandler()
        terminal_formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
        )
        terminal_handler.setFormatter(terminal_formatter)
        logger.addHandler(terminal_handler)

        # Asynchronous file handler setup for non-blocking file logging
        loop = asyncio.get_event_loop()
        async_file_handler = AsyncFileHandler(
            "application_async.log", mode="a", loop=loop
        )
        async_file_formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
        )
        async_file_handler.setFormatter(async_file_formatter)
        logger.addHandler(async_file_handler)

        # Rotating file handler setup for archiving logs
        rotating_file_handler = logging.handlers.RotatingFileHandler(
            "application.log", maxBytes=1048576, backupCount=5
        )
        rotating_file_formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
        )
        rotating_file_handler.setFormatter(rotating_file_formatter)
        logger.addHandler(rotating_file_handler)

        logging.info(
            "Logging setup complete with terminal, asynchronous, and rotating file handlers. Resource profiling initiated."
        )

    async def _initialize_resource_profiling(self):
        """
        Initializes resource profiling if enabled. This method sets up the ResourceProfiler to log resource usage
        at specified intervals asynchronously, ensuring that the profiling does not block or interfere with the
        execution of the program.
        """
        if self.enable_resource_profiling:
            self.resource_profiler = ResourceProfiler(
                interval=self.resource_profiling_interval,
                output=self.resource_profiling_output,
            )
            await self.resource_profiler.start()
            logging.debug("Resource profiling initialized and started.")

    async def _initialize_async_components(self):
        """
        Initializes asynchronous components of the StandardDecorator, including performance logging and resource profiling.
        This method ensures that all asynchronous initializations are completed before the decorator is used.
        """
        await self.initialize_performance_logger()
        await self._initialize_resource_profiling()

    async def _initialize_cache(self):
        """
        Initializes caching mechanisms based on configuration asynchronously.
        """
        if self.enable_caching:
            # Initialize in-memory cache
            self.cache = {}
            # Check and initialize file-based cache if specified
            if os.path.exists(self.file_cache_path):
                async with aiofiles.open(self.file_cache_path, "rb") as f:
                    self.cache = await f.read()
                    self.cache = pickle.loads(self.cache)
            self.logger.debug("Caching mechanisms initialized asynchronously.")

    async def background_cache_persistence(self, key, value):
        """
        Initiates an asynchronous task to persist cache updates to the file cache asynchronously.
        This method is used to ensure non-blocking operations and thread safety in an asyncio context.
        """
        try:
            await asyncio.create_task(self._save_to_file_cache_async(key, value))
        except Exception as e:
            logging.error(
                f"Error initiating background cache persistence for key {key}: {e}"
            )

    def generate_cache_key(
        self, func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """
        Generates a unique cache key based on the function name, arguments, and keyword arguments.
        This method serializes the arguments and keyword arguments to ensure uniqueness.
        """
        sorted_kwargs = tuple(sorted(kwargs.items()))
        serialized_args = pickle.dumps(args)
        serialized_kwargs = pickle.dumps(sorted_kwargs)
        cache_key = (func.__name__, serialized_args, serialized_kwargs)
        logging.debug(f"Generated cache key: {cache_key} for function {func.__name__}")
        return cache_key

    async def cache_logic(
        self, key: Tuple[Any, ...], func: Callable, *args, **kwargs
    ) -> Any:
        """
        Implements the caching logic, checking for cache hits in both in-memory and file caches.
        If a cache miss occurs, the function is executed and the result is cached.
        This method handles both synchronous and asynchronous functions dynamically.
        """
        async with self.cache_lock:
            try:
                # Check in-memory cache first
                if key in self.cache:
                    # Move the key to the end to mark it as recently used
                    self.cache.move_to_end(key)
                    self.call_counter[key] += 1
                    logging.debug(f"Cache hit for {key}.")
                    return self.cache[key][
                        0
                    ]  # Return the result part of the cache entry
                # Check file cache next
                elif key in self.file_cache:
                    result = self.file_cache[key]
                    logging.debug(f"File cache hit for {key}.")
                else:
                    # Handle cache miss
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)
                    logging.debug(
                        f"Cache miss for {key}. Function executed and result cached."
                    )
                    # Update in-memory cache
                    self.cache[key] = (
                        result,
                        datetime.utcnow() + timedelta(seconds=self.cache_lifetime),
                    )
                    self.call_counter[key] = 1
                    # Evict least recently used item if cache exceeds max size
                    if len(self.cache) > self.cache_maxsize:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                        del self.call_counter[oldest_key]
                        logging.debug(
                            f"Evicted least recently used cache item: {oldest_key}"
                        )
                    # Save cache updates to file asynchronously
                    await self.background_cache_persistence(key, self.cache[key])

                    logging.debug(f"Result cached for key: {key}")
                    return result
            except Exception as e:
                logging.error(f"Error in cache logic for key {key}: {e}")
                raise

    async def _load_file_cache(self) -> None:
        """
        Asynchronously loads the file cache from the specified file cache path, ensuring non-blocking I/O operations.
        This method checks the existence of the file cache path and loads the cache using asynchronous file operations,
        thereby preventing the blocking of the event loop and maintaining the responsiveness of the application.

        The method employs a try-except block to gracefully handle any exceptions that may arise during the file
        operations, logging the error details for troubleshooting while ensuring the robustness of the cache loading process.

        Raises:
            Exception: Logs an error message indicating the failure to load the file cache due to an encountered exception.
        """
        try:
            if os.path.exists(self.file_cache_path):
                async with aiofiles.open(self.file_cache_path, mode="rb") as file:
                    file_content = await file.read()
                    # Utilizing asyncio.to_thread to offload the blocking operation, pickle.loads, to a separate thread.
                    self.file_cache = await asyncio.to_thread(
                        pickle.loads, file_content
                    )
                    logging.debug(
                        f"File cache successfully loaded from {self.file_cache_path}."
                    )
            else:
                # Initializing an empty cache if the file cache does not exist.
                self.file_cache = {}
                logging.debug("File cache not found. Initialized an empty cache.")
        except Exception as error:
            # Logging the exception details in case of failure to load the file cache.
            logging.error(f"Failed to load file cache due to error: {error}")
            raise

    async def _save_file_cache(self) -> None:
        """
        Asynchronously saves the current state of the file cache to disk, leveraging asynchronous file operations
        to prevent event loop blocking. This method encapsulates the file writing operation within a try-except block
        to gracefully handle potential exceptions, ensuring the application's robustness and reliability.

        The method utilizes the asyncio.to_thread function to offload the blocking operation, pickle.dump, to a separate
        thread, thereby maintaining the responsiveness of the application by not blocking the event loop.

        Raises:
            Exception: Logs an error message indicating the failure to save the file cache due to an encountered exception.
        """
        try:
            async with aiofiles.open(self.file_cache_path, mode="wb") as file:
                # Asynchronously saving the file cache using the highest protocol available for pickle.
                await asyncio.to_thread(
                    pickle.dump, self.file_cache, file, pickle.HIGHEST_PROTOCOL
                )
                logging.debug("File cache has been successfully saved to disk.")
        except Exception as error:
            # Logging the exception details in case of failure to save the file cache.
            logging.error(f"Failed to save file cache due to error: {error}")
            raise

    async def clean_expired_cache_entries(self):
        """
        Cleans expired entries from both in-memory and file caches asynchronously.
        This method iterates over the cache entries and removes those that have expired.
        """
        try:
            current_time = datetime.utcnow()
            expired_keys = [
                key
                for key, (_, expiration_time) in self.cache.items()
                if expiration_time < current_time
            ]
            for key in expired_keys:
                del self.cache[key]
                logging.debug(f"Removed expired cache entry: {key}")

            expired_keys_file_cache = [
                key
                for key, (_, expiration_time) in self.file_cache.items()
                if expiration_time < current_time
            ]
            for key in expired_keys_file_cache:
                del self.file_cache[key]
            if expired_keys_file_cache:
                await self._save_file_cache()
            logging.debug(
                f"Expired entries removed from file cache: {expired_keys_file_cache}"
            )
        except Exception as e:
            logging.error(f"Error cleaning expired cache entries: {e}")

    async def _evict_lru_from_memory(self):
        """
        Asynchronously evicts the least recently used (LRU) item from the in-memory cache.
        This method is invoked when the cache size exceeds its maximum allowable size, ensuring
        optimal memory usage and cache management. It leverages asynchronous programming paradigms
        to perform non-blocking cache eviction, thereby maintaining the responsiveness and efficiency
        of the application.

        The method employs a sophisticated approach to identify and remove the least recently used item,
        which is based on the assumption that the first item in the cache is the least recently used due
        to the insertion order maintained. This assumption holds true under the condition that the cache
        implementation follows the First-In-First-Out (FIFO) principle for item eviction.

        Raises:
            Exception: Captures and logs any exceptions that occur during the eviction process, ensuring
                       that the method's execution is robust and fault-tolerant. This error handling strategy
                       prevents the propagation of exceptions, which could potentially disrupt the normal
                       operation of the application.
        """
        try:
            # Check if the current cache size exceeds the maximum allowed size
            if len(self.cache) > self.cache_maxsize:
                # Asynchronously remove the least recently used item from the cache
                # The `popitem` method with `last=False` parameter ensures that the first item
                # inserted (and thus the least recently used) is removed
                lru_key, _ = await asyncio.to_thread(self.cache.popitem, last=False)

                # Log the eviction of the least recently used item for monitoring and auditing purposes
                logging.debug(f"LRU item evicted from in-memory cache: {lru_key}")
        except Exception as e:
            # Log any exceptions encountered during the eviction process to facilitate debugging
            # and ensure that the application can gracefully recover from unexpected errors
            logging.error(f"Error evicting LRU item from memory: {e}")

    async def maintain_cache(self):
        """
        Periodically checks and maintains the cache by cleaning expired entries and evicting LRU items.
        This method runs in a separate asynchronous task to ensure efficient cache management.
        """
        try:
            # This enhanced block continuously monitors and maintains the cache, incorporating advanced logic to
            # determine the program's execution context and initiate a graceful shutdown if the cache remains unchanged
            # for an extended period, indicating potential program inactivity or improper closure.

            # Initialize variables to track cache changes and manage shutdown logic
            unchanged_cycles = (
                0  # Counter for the number of cycles with no cache changes
            )
            is_main_program = (
                __name__ == "__main__"
            )  # Check if running as the main program
            sleep_duration = (
                60 if is_main_program else 10
            )  # Sleep duration based on execution context

            # Continuously perform cache maintenance tasks with advanced monitoring for inactivity
            while True:
                await asyncio.sleep(
                    sleep_duration
                )  # Pause execution for the determined duration

                # Capture the state of caches before maintenance to detect changes
                initial_memory_cache_state = str(self.cache)
                initial_file_cache_state = str(self.file_cache)

                # Execute cache maintenance operations
                await self.clean_expired_cache_entries()  # Clean expired entries from caches
                self._evict_lru_from_memory()  # Evict least recently used items from memory cache

                # Determine if the cache states have changed following maintenance operations
                memory_cache_changed = initial_memory_cache_state != str(self.cache)
                file_cache_changed = initial_file_cache_state != str(self.file_cache)

                # Log the completion of cache maintenance based on the program's execution context
                context_message = (
                    "the main program"
                    if is_main_program
                    else "the imported module context"
                )
                logging.debug(f"Cache maintenance completed in {context_message}.")

                # Update the unchanged cycles counter based on cache state changes
                if memory_cache_changed or file_cache_changed:
                    unchanged_cycles = 0  # Reset counter if changes were detected
                else:
                    unchanged_cycles += (
                        1  # Increment counter if no changes were detected
                    )

                # Initiate a graceful shutdown if caches have remained unchanged for 10 cycles
                if unchanged_cycles >= 10:
                    logging.info(
                        "No cache changes detected for 10 cycles. Initiating graceful shutdown."
                    )
                    # Placeholder for graceful shutdown logic
                    # This should include tasks such as closing database connections, stopping async tasks,
                    # and any other cleanup required before safely terminating the program.
                    await self.initiate_graceful_shutdown()
                    break  # Exit the loop to stop further cache maintenance

        except Exception as e:
            # Log any exceptions encountered during the cache maintenance and monitoring process
            logging.error(f"Error in cache maintenance task: {e}")

    async def attempt_cache_retrieval(self, key: Tuple[Any, ...]) -> Optional[Any]:
        """
        Attempts to retrieve the cached result for the given key from both in-memory and file caches.
        This method checks the caches and returns the cached result if found.
        """
        try:
            if key in self.cache:
                logging.debug(f"In-memory cache hit for {key}")
                return self.cache[key]
            elif key in self.file_cache:
                logging.debug(f"File cache hit for {key}")
                result = self.file_cache[key]
                if len(self.cache) >= self.cache_maxsize:
                    self.cache.popitem(last=False)
                self.cache[key] = result
                return result
            return None
        except Exception as e:
            logging.error(f"Error attempting cache retrieval for key {key}: {e}")
            return None

    async def update_cache(
        self, key: Tuple[Any, ...], result: Any, is_async: bool = True
    ):
        """
        Updates the cache with the given key and result, managing cache sizes, evictions, and time-based expiration.
        This method handles both in-memory and file cache updates, including time-based eviction.
        """
        try:
            current_time = datetime.utcnow()
            expiration_time = current_time + timedelta(seconds=self.cache_lifetime)
            if len(self.cache) >= self.cache_maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = (result, expiration_time)
            if self.call_counter.get(key, 0) >= self.call_threshold:
                self.call_counter[key] = 0
                if is_async:
                    await self._save_to_file_cache(key, (result, expiration_time))
            else:
                self.background_cache_persistence(key, (result, expiration_time))
                logging.debug(
                    f"Cache updated for key: {key}. Cache size: {len(self.cache)}"
                )
        except Exception as e:
            logging.error(f"Error updating cache for key {key}: {e}")

    async def _save_to_file_cache(
        self, key: Tuple[Any, ...], value: Tuple[Any, datetime]
    ):
        """
        Asynchronously saves a key-value pair to the file cache.
        This method uses asynchronous file operations to ensure the event loop is not blocked.
        """
        try:
            self.file_cache[key] = value
            async with aiofiles.open(self.file_cache_path, "wb") as f:
                await asyncio.to_thread(
                    pickle.dump, self.file_cache, f, pickle.HIGHEST_PROTOCOL
                )
            logging.debug(f"File cache asynchronously updated for key: {key}")
        except Exception as e:
            logging.error(
                f"Error saving to file cache asynchronously for key {key}: {e}"
            )

    def _save_to_file_cache_sync(
        self, key: Tuple[Any, ...], value: Tuple[Any, datetime]
    ):
        """
        Synchronously saves a key-value pair to the file cache.
        This method is used for background cache persistence where asynchronous operations are not feasible.
        """
        try:
            self.file_cache[key] = value
            with open(self.file_cache_path, "wb") as f:
                pickle.dump(self.file_cache, f, pickle.HIGHEST_PROTOCOL)
            logging.debug(f"File cache synchronously updated for key: {key}")
        except Exception as e:
            logging.error(
                f"Error saving to file cache synchronously for key {key}: {e}"
            )

    def __call__(self, func: F) -> F:
        """
        Makes the class instance callable, allowing it to be used as a decorator. It wraps the
        decorated function in a new function that performs argument validation, caching, logging,
        retry logic, and performance monitoring before executing the original function.

        This method dynamically adapts to both synchronous and asynchronous functions, ensuring
        that all enhanced functionalities are applied consistently across different types of function
        executions. It leverages internal methods for argument validation, caching logic, retry mechanisms,
        and performance logging to provide a comprehensive enhancement to the decorated function.

        Args:
            func (F): The function to be decorated.

        Returns:
            F: The wrapped function, which includes enhanced functionality.
        """

        async def run_async_coroutine(coroutine):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    return asyncio.ensure_future(coroutine, loop=loop)
            except RuntimeError:
                return asyncio.run(coroutine)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                logging.info(
                    f"Async call to {func.__name__} with args: {args} and kwargs: {kwargs}"
                )
                if self.enable_validation:
                    await self._validate_func_signature(func, *args, **kwargs)
                try:
                    start_time = time.perf_counter()
                    result = await self.wrapper_logic(func, True, *args, **kwargs)
                    end_time = time.perf_counter()
                    await self.log_performance(func, start_time, end_time)
                    return result
                except Exception as e:
                    logging.error(f"Exception in async call to {func.__name__}: {e}")
                    raise
                finally:
                    self.end_time = time.perf_counter()

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                logging.info(
                    f"Sync call to {func.__name__} with args: {args} and kwargs: {kwargs}"
                )
                if self.enable_validation:
                    asyncio.run_coroutine_threadsafe(
                        self._validate_func_signature(func, *args, **kwargs),
                        asyncio.get_event_loop(),
                    )
                try:
                    start_time = time.perf_counter()
                    result = asyncio.run_coroutine_threadsafe(
                        self.wrapper_logic(func, False, *args, **kwargs),
                        asyncio.get_event_loop(),
                    ).result()
                    end_time = time.perf_counter()
                    asyncio.run_coroutine_threadsafe(
                        self.log_performance(func, start_time, end_time),
                        asyncio.get_event_loop(),
                    )
                    return result
                except Exception as e:
                    logging.error(f"Exception in sync call to {func.__name__}: {e}")
                    raise

            return sync_wrapper

    async def wrapper_logic(self, func: F, is_async: bool, *args, **kwargs) -> Any:
        """
        Contains the core logic for retrying, caching, logging, validating, and monitoring the execution
        of the decorated function. This method dynamically adapts to both synchronous and asynchronous functions,
        ensuring that the execution logic is seamlessly applied regardless of the function's nature.

        The method's functionality includes:
        - Argument validation to ensure compliance with specified criteria.
        - Caching logic for efficient retrieval of function results, minimizing execution time for repeated calls.
        - Retry mechanisms to address transient failures by re-executing the function according to predefined rules.
        - Performance logging for insights into execution efficiency.

        Args:
            func (F): The function to be executed.
            is_async (bool): Indicates if the function is asynchronous.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function execution, either from cache or newly computed.
        """
        # Initialize performance monitoring
        start_time = time.perf_counter()
        cache_key = self.generate_cache_key(
            func, args, kwargs
        )  # Assuming generate_cache_key method exists

        # Ensure thread safety with asyncio.Lock for cache operations
        async with self.cache_lock:
            # Cache logic initialization
            if self.enable_caching:
                cache_key = self.cache_key_strategy(func, args, kwargs)
                cached_result = await self.attempt_cache_retrieval(cache_key)
                if cached_result is not None:
                    logging.info(f"Cache hit for {func.__name__} with key {cache_key}")
                    return cached_result
                else:
                    logging.info(f"Cache miss for {func.__name__} with key {cache_key}")

            # Retry Logic
            if self.dynamic_retry_enabled:
                retries, delay = self.dynamic_retry_strategy(Exception)

            # Initialize retry attempt counter
            attempt = 0
            while attempt <= self.retries:
                try:
                    if is_async:
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    # Cache the result if caching is enabled
                    if self.enable_caching:
                        await self.update_cache(cache_key, result, is_async)
                        logging.info(
                            f"Result cached for {func.__name__} with key {cache_key}"
                        )

                    return result
                except self.retry_exceptions as e:
                    logging.warning(
                        f"Retry {attempt + 1} for {func.__name__} due to {e}"
                    )
                    if attempt < self.retries:
                        if is_async:
                            await asyncio.sleep(delay)
                        else:
                            time.sleep(delay)
                    attempt += 1
                except Exception as e:
                    logging.error(f"Exception during {func.__name__} execution: {e}")
                    raise e
                finally:
                    # Performance monitoring
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    logging.info(
                        f"Execution of {func.__name__} completed in {execution_time:.2f}s"
                    )
                    if is_async:
                        await self.log_performance(func, start_time, end_time)
                    else:
                        # Synchronous logging of performance
                        self.log_performance_sync(func, start_time, end_time)

    async def _validate_func_signature(self, func, *args, **kwargs):
        """
        Validates the function's signature against provided arguments and types.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions.
        It leverages Python's introspection capabilities to validate function signatures dynamically.

        Args:
            func (Callable): The function whose signature is being validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If the provided arguments do not match the function's signature.

        Returns:
            None: This method does not return any value but raises an exception on failure.
        """
        # Capture the start time for performance logging
        start_time = time.perf_counter()
        logging.debug(
            f"Validating function signature for {func.__name__} at {start_time}"
        )

        # Retrieve the function's signature and bind the provided arguments
        sig = signature(func)
        logging.debug(f"Function signature: {sig}")
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        logging.debug(f"Bound arguments with defaults applied: {bound_args}")

        # Retrieve type hints and validate each argument against its expected type
        type_hints = get_type_hints(func)
        logging.debug(f"Type hints: {type_hints}")
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name, None)  # Set a default value of None
            logging.debug(
                f"Validating argument '{name}' with value '{value}' against expected type '{expected_type}'"
            )
            if expected_type is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
                )

        # Log the completion of the validation process
        end_time = time.perf_counter()
        logging.debug(
            f"Validation of function signature for {func.__name__} completed in {end_time - start_time:.2f}s"
        )

    async def _get_arg_position(self, func: F, arg_name: str) -> int:
        """
        Determines the position of an argument in the function's signature.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions.
        It leverages Python's introspection capabilities to dynamically determine argument positions.

        Args:
            func (Callable): The function being inspected.
            arg_name (str): The name of the argument whose position is sought.

        Returns:
            int: The position of the argument in the function's signature.

        Raises:
            ValueError: If the argument name is not found in the function's signature.
        """
        # Capture the start time for performance logging
        start_time = time.perf_counter()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} at {start_time}"
        )

        # Determine the position of the argument in the function's signature
        parameters = list(signature(func).parameters)
        if arg_name not in parameters:
            raise ValueError(
                f"Argument '{arg_name}' not found in {func.__name__}'s signature"
            )
        result = parameters.index(arg_name)

        # Log the determined position
        logging.debug(f"Argument position for {arg_name} in {func.__name__}: {result}")

        # Log the completion of the process
        end_time = time.perf_counter()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} completed in {end_time - start_time:.2f}s"
        )
        return result

    async def validate_arguments(self, func: F, *args, **kwargs) -> None:
        """
        Validates the arguments passed to a function against expected type hints and custom validation rules.
        Adjusts for whether the function is a bound method (instance or class method) or a regular function
        or static method, and applies argument validation accordingly.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions,
        leveraging asyncio for non-blocking operations and ensuring thread safety with asyncio.Lock.

        Args:
            func (Callable): The function whose arguments are to be validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If an argument does not match its expected type.
            ValueError: If an argument fails custom validation rules.
        """
        # Initialize performance logging
        start_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validating arguments for {func.__name__} at {start_time} with args: {args} and kwargs: {kwargs}",
            extra={"async_mode": True},
        )

        # Adjust args for bound methods (instance or class methods)
        if inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        ):
            # For bound methods, the first argument ('self' or 'cls') should not be included in validation
            args = args[1:]

        # Attempt to bind args and kwargs to the function's signature
        try:
            bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
        except TypeError as e:
            logging.error(
                f"Error binding arguments for {func.__name__}: {e}",
                extra={"async_mode": True},
            )
            raise

        bound_args.apply_defaults()
        type_hints = get_type_hints(func)

        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name)
            if not await self.validate_type(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type '{expected_type}', got type '{type(value)}'"
                )

            validation_rule = self.validation_rules.get(name)
            if validation_rule:
                valid = (
                    await validation_rule(value)
                    if asyncio.iscoroutinefunction(validation_rule)
                    else validation_rule(value)
                )
                if not valid:
                    raise ValueError(
                        f"Validation failed for argument '{name}' with value '{value}'"
                    )

        end_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validation completed at {end_time} taking total time of {end_time - start_time} seconds",
            extra={"async_mode": True},
        )

    async def validate_type(self, value: Any, expected_type: Any) -> bool:
        """
        Recursively validates a value against an expected type, handling generics, special forms, and complex types.
        This method is meticulously designed to be exhaustive in its approach to type validation,
        ensuring compatibility with a wide range of type annotations, including generics, special forms, and complex types.
        It leverages Python's typing module to interpret and validate against the provided type hints accurately.
        Utilizes asyncio for non-blocking operations and ensures thread safety with asyncio.Lock.

        Args:
            value (Any): The value to validate.
            expected_type (Any): The expected type against which to validate the value.

        Returns:
            bool: True if the value matches the expected type, False otherwise.
        """
        start_time = asyncio.get_event_loop().time()
        # Early exit for typing.Any, indicating any type is acceptable.
        if expected_type is Any:
            logging.debug(
                "Any type encountered, validation passed.", extra={"async_mode": True}
            )
            return True

        # Handle Union types, including Optional, by validating against each type argument until one matches.
        if get_origin(expected_type) is Union:
            logging.debug(
                f"Union type encountered: {expected_type}", extra={"async_mode": True}
            )
            return any(
                await self.validate_type(value, arg) for arg in get_args(expected_type)
            )

        # Handle special forms like Any, ClassVar, etc., assuming validation passes for these.
        if isinstance(expected_type, _SpecialForm):
            logging.debug(
                f"Special form encountered: {expected_type}", extra={"async_mode": True}
            )
            return True

        # Extract the origin type and type arguments from the expected type, if applicable.
        origin_type = get_origin(expected_type)
        type_args = get_args(expected_type)

        # Handle generic types (List[int], Dict[str, Any], etc.)
        if origin_type is not None:
            if not isinstance(value, origin_type):
                logging.debug(
                    f"Value {value} does not match the origin type {origin_type}.",
                    extra={"async_mode": True},
                )
                return False
            if type_args:
                # Validate type arguments (e.g., the 'int' in List[int])
                if issubclass(origin_type, collections.abc.Mapping):
                    key_type, val_type = type_args
                    logging.debug(
                        f"Validating Mapping with key type {key_type} and value type {val_type}.",
                        extra={"async_mode": True},
                    )
                    return all(
                        await self.validate_type(k, key_type)
                        and await self.validate_type(v, val_type)
                        for k, v in value.items()
                    )
                elif issubclass(
                    origin_type, collections.abc.Iterable
                ) and not issubclass(origin_type, (str, bytes, bytearray)):
                    element_type = type_args[0]
                    logging.debug(
                        f"Validating each element in Iterable against type {element_type}.",
                        extra={"async_mode": True},
                    )
                    return all(
                        [await self.validate_type(elem, element_type) for elem in value]
                    )
                # Extend to handle other generic types as needed
        else:
            # Handle non-generic types directly
            if not isinstance(value, expected_type):
                logging.debug(
                    f"Value {value} does not match the expected non-generic type {expected_type}.",
                    extra={"async_mode": True},
                )
                return False
            return True

        # Fallback for unsupported types
        logging.debug(
            f"Type {expected_type} not supported by the current validation logic.",
            extra={"async_mode": True},
        )
        return False
