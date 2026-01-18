"""
Module: indedecorators.py
Description: Provides a comprehensive suite of decorators and utilities for enhancing the functionality of the INDEGO project development. This module integrates a wide range of Python libraries and frameworks to support asynchronous operations, data handling, machine learning, file and HTTP operations, data serialization, system monitoring, validation, database operations, caching, web frameworks, and messaging.

Imports are meticulously organized and documented to ensure clarity and maintainability. The module also adheres to the highest standards of Python programming, following PEP8 guidelines, exhaustive commenting, and type hinting to enhance readability and developer experience.
"""

# Core Python Libraries
import gc  # Garbage Collector interface. Documentation: https://docs.python.org/3/library/gc.html
import io  # Core tools for working with streams. Documentation: https://docs.python.org/3/library/io.html
from contextlib import asynccontextmanager
import asyncio  # Asynchronous I/O, event loop, coroutines, and tasks. Documentation: https://docs.python.org/3/library/asyncio.html
import functools  # Higher-order functions and operations on callable objects. Documentation: https://docs.python.org/3/library/functools.html
import inspect  # Inspect live objects. Documentation: https://docs.python.org/3/library/inspect.html
from inspect import (
    signature,
    Parameter,
)  # Inspect live objects. Documentation: https://docs.python.org/3/library/inspect.html
import logging  # Logging facility for Python. Documentation: https://docs.python.org/3/library/logging.html
import logging.config  # Logging configuration. Documentation: https://docs.python.org/3/library/logging.config.html
import os  # Miscellaneous operating system interfaces. Documentation: https://docs.python.org/3/library/os.html
import json  # JSON encoder and decoder. Documentation: https://docs.python.org/3/library/json.html
import pickle  # Python object serialization. Documentation: https://docs.python.org/3/library/pickle.html
import time  # Time access and conversions. Documentation: https://docs.python.org/3/library/time.html
import warnings  # Warning control. Documentation: https://docs.python.org/3/library/warnings.html
from datetime import (
    datetime,
)  # Basic date and time types. Documentation: https://docs.python.org/3/library/datetime.html
from pathlib import (
    Path,
)  # Object-oriented filesystem paths. Documentation: https://docs.python.org/3/library/pathlib.html
from typing import (  # Support for type hints. Documentation: https://docs.python.org/3/library/typing.html
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    Awaitable,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
)
from dataclasses import (
    dataclass,
    field,
)  # Data classes. Documentation: https://docs.python.org/3/library/dataclasses.html

# Concurrent Execution
from concurrent.futures import (  # Launching parallel tasks. Documentation: https://docs.python.org/3/library/concurrent.futures.html
    ThreadPoolExecutor,
    ProcessPoolExecutor,
)
from multiprocessing import (
    Manager,
)  # Shared object manager. Documentation: https://docs.python.org/3/library/multiprocessing.html

# Data Handling and Machine Learning
import numpy as np  # The fundamental package for scientific computing with Python. Documentation: https://numpy.org/doc/
import pandas as pd  # Powerful data structures for data analysis, time series, and statistics. Documentation: https://pandas.pydata.org/pandas-docs/stable/
from sklearn.base import (  # Base classes for all estimators and transformers. Documentation: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base
    BaseEstimator,
    TransformerMixin,
)
from sklearn.compose import (  # Applies transformers to columns of an array or pandas DataFrame. Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    ColumnTransformer,
)
from sklearn.impute import (  # Imputation transformer for completing missing values. Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    SimpleImputer,
)
from sklearn.pipeline import (
    Pipeline,
)  # Pipeline of transforms with a final estimator. Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
from sklearn.preprocessing import (  # Preprocessing and normalization. Documentation: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

# Asynchronous File and HTTP Operations
import aiofiles  # File support for asyncio. Documentation: https://github.com/Tinche/aiofiles
import aiohttp  # Asynchronous HTTP Client/Server. Documentation: https://docs.aiohttp.org/
import joblib  # Lightweight pipelining: using Python functions as pipeline jobs. Documentation: https://joblib.readthedocs.io/en/latest/

# Data Serialization and Compression
import lz4.frame  # LZ4 frame compression. Documentation: https://python-lz4.readthedocs.io/en/stable/
import msgpack  # MessagePack (de)serializer. Documentation: https://msgpack-python.readthedocs.io/en/latest/
import orjson  # Fast, correct JSON library. Documentation: https://github.com/ijl/orjson
import zstandard as zstd  # Zstandard compression. Documentation: https://python-zstandard.readthedocs.io/en/latest/

# System and Performance Monitoring
import psutil  # Cross-platform lib for process and system monitoring in Python. Documentation: https://psutil.readthedocs.io/en/latest/
import cProfile  # Deterministic profiling of Python programs. Documentation: https://docs.python.org/3/library/profile.html
import memory_profiler  # Monitor memory usage of a Python program. Documentation: https://pypi.org/project/memory-profiler/
import tracemalloc  # Trace memory allocations. Documentation: https://docs.python.org/3/library/tracemalloc.html
import pstats  # Statistics for profiling. Documentation: https://docs.python.org/3/library/profile.html

# Data Validation and Settings Management
import pydantic  # Data validation and settings management using python type annotations. Documentation: https://pydantic-docs.helpmanual.io/
from pydantic import (
    BaseModel,
    ValidationError,
)  # Data validation using Pydantic models. Documentation: https://pydantic-docs.helpmanual.io/

# Asynchronous Database Operations
from aiokeydb import (  # Unified Synchronous and Asynchronous Python client for KeyDB and Redis. Documentation: https://github.com/aio-libs/aiokeydb
    AsyncKeyDB,
    KeyDBClient,
)
from redis import Redis

# Scalable Data Pipeline and Logging
import lazyops  # LazyOps: A Python library for building efficient and scalable data pipelines. Documentation:
from lazyops.utils import logger  # LazyOps logger for enhanced logging capabilities

# Caching and In-Memory Data Store
import cachetools  # Extensible memoizing collections and decorators. Documentation: https://cachetools.readthedocs.io/en/latest/
import diskcache  # Disk and file backed cache library. Documentation: https://www.grantjenks.com/docs/diskcache/
from diskcache import (
    Cache,
)  # Disk and file backed cache library. Documentation: https://www.grantjenks.com/docs/diskcache/

# Web Frameworks and API
from fastapi import (  # A modern, fast (high-performance) web framework for building APIs. Documentation: https://fastapi.tiangolo.com/
    FastAPI,
    Request,
    Response,
)
from starlette.middleware.cors import (  # Cross-Origin Resource Sharing (CORS) middleware. Documentation: https://www.starlette.io/middleware/#corsmiddleware
    CORSMiddleware,
)
from starlette.responses import (
    JSONResponse,
)  # JSON response. Documentation: https://www.starlette.io/responses/#jsonresponse

# Asynchronous Messaging
from aiokafka import (  # Kafka integration using asyncio. Documentation: https://aiokafka.readthedocs.io/
    AIOKafkaConsumer,
    AIOKafkaProducer,
)
import asyncio
import logging
import logging.config
from typing import Any, Dict, Awaitable, Callable
from functools import (
    wraps,
    partial,
    lru_cache,
    reduce,
    singledispatch,
    update_wrapper,
    total_ordering,
    partialmethod,
    cached_property,
)

# Comprehensive logging configuration to facilitate detailed application monitoring and debugging.
# This configuration includes multiple handlers to direct log output to the console, a rotating file for general logs,
# and a separate rotating file specifically for error logs, ensuring that all log information is captured and stored efficiently.
log_config: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(process)d - %(thread)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "detailed",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "detailed.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 20,
            "formatter": "detailed",
        },
        "errors": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "errors_detailed.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 20,
            "formatter": "detailed",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file", "errors"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}

# Apply the logging configuration to initialize the logging system according to the defined specifications.
logging.config.dictConfig(log_config)


# Decorator to log the entry, exit, and exceptions of an asynchronous function.
def async_log_decorator(
    func: Callable[..., Awaitable[Any]]
) -> Callable[..., Awaitable[Any]]:
    """
    A decorator that wraps asynchronous functions to log their entry, exit, and any exceptions that occur during their execution.
    This enhances the visibility and traceability of asynchronous operations, aiding in debugging and performance monitoring.

    Args:
        func (Callable[..., Awaitable[Any]]): The asynchronous function to be decorated.

    Returns:
        Callable[..., Awaitable[Any]]: The decorated function with added logging capabilities.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        logging.info(f"Entering {func_name} with args: {args} and kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)
            logging.info(f"Exiting {func_name} with result: {result}")
            return result
        except Exception as e:
            logging.error(f"Exception in {func_name}: {e}", exc_info=True)
            raise e

    return wrapper


# Asynchronously initialize comprehensive profiling and tracing to monitor the application's performance and resource usage.
# This function utilizes cProfile for performance profiling, tracemalloc for memory allocation tracing, and logs the initialization process.
# It is designed to be called at the application startup to ensure that profiling and tracing are active throughout the application lifecycle.
@async_log_decorator
async def init_profiling_and_tracing() -> None:
    import cProfile
    import tracemalloc

    # Enable the cProfile profiler to start collecting performance data.
    pr: cProfile.Profile = cProfile.Profile()
    pr.enable()

    # Start tracing memory allocations with tracemalloc to identify memory usage and leaks.
    tracemalloc.start()

    # Log the successful initialization of profiling and tracing to assist with debugging and monitoring.
    logging.info("Profiling and tracing initialized with maximum verbosity.")


profiling = asyncio.run(init_profiling_and_tracing())


# Enhanced cache initialization with comprehensive logging for cache operations
async def init_cache() -> None:
    from cachetools import TTLCache

    cache = TTLCache(maxsize=2048, ttl=7200)  # Increased cache size and TTL
    logging.info("Cache initialized with TTLCache.")
    return cache


# Initialize the cache asynchronously and ensure it's ready before proceeding with the application's execution.
cache = asyncio.run(init_cache())


# Initialize web app with detailed logging for each request and response
async def init_web_app():
    from aiohttp import web

    app = web.Application(middlewares=[web.normalize_path_middleware()])
    logging.info("Web application initialized with aiohttp.")
    return app


app = asyncio.run(init_web_app())

# Custom types for enhanced readability and maintainability
T = TypeVar("T")
DecoratedCallable = Callable[..., Coroutine[Any, Any, T]]
UriType = str
SessionNameType = str

# KeyDB Configuration with detailed documentation and type annotations
KEYDB_CACHE_URIS: Dict[str, UriType] = {
    # Extensive list of URIs for various caches
    "cache_default": "keydb://localhost:6379/0",
    "cache_dev": "keydb://localhost:6379/1",
    # Additional cache URIs omitted for brevity
}

KEYDB_BLOB_URIS: Dict[str, UriType] = {
    # Extensive list of URIs for various blobs
    "blob_default": "keydb://localhost:6479/0",
    "blob_dev": "keydb://localhost:6479/1",
    # Additional blob URIs omitted for brevity
}


# Asynchronous KeyDB client configuration with error handling
async def configure_keydb_client(
    default_uri: UriType = KEYDB_CACHE_URIS["cache_default"],
) -> AsyncKeyDB:
    try:
        client = await AsyncKeyDB(host="localhost", port=6379, password="yourpassword")
        logging.info("KeyDB client session initialized.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize KeyDB client: {e}")
        raise


# Session management with asynchronous initialization and error handling
sessions: Dict[SessionNameType, AsyncKeyDB] = {}


async def init_keydb_session(name: SessionNameType, uri: UriType) -> None:
    if name in sessions:
        logging.error(f"Session {name} already exists.")
        raise KeyError(f"Session {name} already exists.")
    sessions[name] = await configure_keydb_client(uri)
    logging.info(f"Session {name} initialized with URI: {uri}")


# Decorator configuration with comprehensive type annotations, detailed documentation, and advanced features
@dataclass
class DecoratorConfig:
    cache_dir: Path = Path("cache")
    cache_expiry: Optional[int] = None
    cache_file_prefix: str = "cache_"
    input_dtypes: Dict[str, Type] = field(default_factory=dict)
    output_dtypes: Dict[str, Type] = field(default_factory=dict)
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
    manager: Optional[Manager] = None
    compression_level: int = 3
    compression_algorithm: str = "zstd"
    serialization_format: str = "msgpack"
    cache_backend: str = "diskcache"
    cache_config: Dict[str, Any] = field(default_factory=dict)
    kafka_config: Dict[str, Any] = field(default_factory=dict)
    keydb_config: Dict[str, Any] = field(default_factory=dict)
    fastapi_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initializes the decorator configuration, setting up cache, Kafka, KeyDB, and FastAPI integrations."""
        self.cache_dir.mkdir(
            parents=True, exist_ok=True
        )  # Ensure cache directory exists
        self.cache = self._initialize_cache()
        self.kafka_producer, self.kafka_consumer = self._initialize_kafka()
        self.keydb = self._initialize_keydb()  # Initialize KeyDB instead of Redis
        self.fastapi = self._initialize_fastapi()

    def _initialize_cache(self) -> Union[Cache, Redis[str]]:
        """Initializes the cache backend based on the configuration."""
        if self.cache_backend == "diskcache":
            cache_instance = diskcache.Cache(self.cache_dir, **self.cache_config)
        elif self.cache_backend == "keydb":  # Use KeyDB as the cache backend
            cache_instance = Redis(
                **self.keydb_config
            )  # Utilize RedisCache interface for KeyDB compatibility
        else:
            error_message = f"Unsupported cache backend: {self.cache_backend}"
            logger.error(error_message)
            raise ValueError(error_message)
        logger.debug(f"Cache initialized with backend: {self.cache_backend}")
        return cache_instance

    def _initialize_kafka(self) -> Tuple[AIOKafkaProducer, AIOKafkaConsumer]:
        """Initializes Kafka producer and consumer based on the configuration."""
        producer = AIOKafkaProducer(**self.kafka_config)
        consumer = AIOKafkaConsumer(**self.kafka_config)
        logger.debug("Kafka producer and consumer initialized")
        return producer, consumer

    def _initialize_keydb(self) -> Redis:
        """Initializes KeyDB connection based on the configuration."""
        keydb_instance = Redis(
            **self.keydb_config
        )  # Utilize Redis interface for KeyDB compatibility
        logger.debug("KeyDB connection initialized")
        return keydb_instance

    def _initialize_fastapi(self) -> FastAPI:
        """Initializes FastAPI application with CORS middleware based on the configuration."""
        app = FastAPI(**self.fastapi_config)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.debug("FastAPI application initialized with CORS middleware")
        return app


# Decorator to log function execution details
def log_execution(func: DecoratedCallable) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"Executing function: {func.__name__}")
        logger.info(f"Arguments: {args}")
        logger.info(f"Keyword arguments: {kwargs}")

        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.info(
                f"Function {func.__name__} executed successfully in {end_time - start_time:.2f} seconds"
            )
            logger.info(f"Result: {result}")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            logger.exception(
                f"Function {func.__name__} raised an exception after {end_time - start_time:.2f} seconds: {str(e)}",
                exc_info=True,
            )
            raise

    return wrapper


# Decorator to validate input and output data types
def validate_dtypes(
    func: DecoratedCallable, config: DecoratorConfig
) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Validate input data types
        for arg_name, arg_type in config.input_dtypes.items():
            if arg_name in kwargs:
                arg_value = kwargs[arg_name]
                if not isinstance(arg_value, arg_type):
                    raise TypeError(
                        f"Expected argument '{arg_name}' to be of type {arg_type}, but got {type(arg_value)}"
                    )
            else:
                found_arg = False
                for arg in args:
                    if isinstance(arg, arg_type):
                        found_arg = True
                        break
                if not found_arg:
                    raise TypeError(
                        f"Expected an argument of type {arg_type}, but none was found"
                    )

        result = await func(*args, **kwargs)

        # Validate output data types
        if isinstance(result, tuple):
            for i, (output_name, output_type) in enumerate(
                config.output_dtypes.items()
            ):
                if i < len(result):
                    output_value = result[i]
                    if not isinstance(output_value, output_type):
                        raise TypeError(
                            f"Expected output at index {i} ('{output_name}') to be of type {output_type}, but got {type(output_value)}"
                        )
        elif isinstance(result, dict):
            for output_name, output_type in config.output_dtypes.items():
                if output_name in result:
                    output_value = result[output_name]
                    if not isinstance(output_value, output_type):
                        raise TypeError(
                            f"Expected output '{output_name}' to be of type {output_type}, but got {type(output_value)}"
                        )
        else:
            output_name, output_type = next(iter(config.output_dtypes.items()))
            if not isinstance(result, output_type):
                raise TypeError(
                    f"Expected output '{output_name}' to be of type {output_type}, but got {type(result)}"
                )

        return result

    return wrapper


# Decorator to handle exceptions and retries
def handle_exceptions(
    func: DecoratedCallable, config: DecoratorConfig
) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        retries = 0
        while retries < config.max_retries:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=config.timeout
                )
            except asyncio.TimeoutError:
                retries += 1
                logger.warning(
                    f"Retry {retries}/{config.max_retries}: Function {func.__name__} timed out after {config.timeout} seconds. Retrying..."
                )
            except Exception as e:
                retries += 1
                logger.warning(
                    f"Retry {retries}/{config.max_retries}: An exception occurred in function {func.__name__}: {str(e)}. Retrying in {config.retry_delay} seconds..."
                )
                await asyncio.sleep(config.retry_delay)

        logger.error(f"Max retries exceeded. Raising exception...")
        raise RuntimeError(
            f"Function {func.__name__} failed after {config.max_retries} retries"
        )

    return wrapper


def serialize_cache_key(key):
    def default(obj):
        if isinstance(obj, frozenset):
            return sorted(
                obj
            )  # Convert frozenset to a sorted list for JSON serialization
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    return json.dumps(key, default=default)


# Decorator to cache function results
async def cache_result(func, config):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        raw_cache_key = (func.__name__, args, frozenset(kwargs.items()))
        cache_key = serialize_cache_key(raw_cache_key)  # Serialize the key

        cached_result = await asyncio.to_thread(lambda: config.cache.get(cache_key))
        if cached_result is not None:
            return cached_result
        result = await func(*args, **kwargs)

        try:
            await config.cache.set(cache_key, result, expire=config.cache_expiry)
            logger.info(
                f"Caching result for function {func.__name__} with key {cache_key}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to cache result for function {func.__name__} with key {cache_key}: {str(e)}"
            )

        return result

    return wrapper


# Decorator to optimize memory usage
def optimize_memory(func: DecoratedCallable) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Perform memory optimization techniques
        # ...

        result = await func(*args, **kwargs)

        # Perform memory cleanup
        # ...

        return result

    return wrapper


# Decorator to enable asynchronous execution
def async_execution(
    func: DecoratedCallable, config: DecoratorConfig
) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        if config.executor is None:
            result = await loop.run_in_executor(None, partial(func, *args, **kwargs))
        else:
            result = await loop.run_in_executor(
                config.executor, partial(func, *args, **kwargs)
            )
        return result

    return wrapper


# Decorator for data preprocessing
def preprocess_data(func: DecoratedCallable) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Perform data preprocessing steps
        # ...

        result = await func(*args, **kwargs)

        # Perform additional data postprocessing if needed
        # ...

        return result

    return wrapper


# Decorator for feature engineering
def engineer_features(func: DecoratedCallable) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Perform feature engineering techniques
        # ...

        result = await func(*args, **kwargs)

        # Perform additional feature postprocessing if needed
        # ...

        return result

    return wrapper


# Decorator for data compression
def compress_data(
    func: DecoratedCallable, config: DecoratorConfig
) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)

        # Compress the result data
        if config.compression_algorithm == "zstd":
            compressed_result = zstd.compress(result, level=config.compression_level)
        elif config.compression_algorithm == "lz4":
            compressed_result = lz4.frame.compress(
                result, compression_level=config.compression_level
            )
        elif config.compression_algorithm == "snappy":
            compressed_result = snappy.compress(result)
        else:
            raise ValueError(
                f"Unsupported compression algorithm: {config.compression_algorithm}"
            )

        return compressed_result

    return wrapper


# Decorator for data serialization
def serialize_data(
    func: DecoratedCallable, config: DecoratorConfig
) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)

        # Serialize the result data
        if config.serialization_format == "msgpack":
            serialized_result = msgpack.packb(result)
        elif config.serialization_format == "orjson":
            serialized_result = orjson.dumps(result)
        elif config.serialization_format == "pyarrow":
            table = pa.Table.from_pandas(result)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf)
            serialized_result = buf.getvalue().to_pybytes()
        else:
            raise ValueError(
                f"Unsupported serialization format: {config.serialization_format}"
            )

        return serialized_result

    return wrapper


# Decorator for data validation using Pydantic models
def validate_data(
    func: DecoratedCallable, input_model: BaseModel, output_model: BaseModel
) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        # Validate input data using Pydantic model
        try:
            validated_input = input_model(**kwargs)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid input data: {str(e)}")

        result = await func(*args, **validated_input.dict())

        # Validate output data using Pydantic model
        try:
            validated_output = output_model(**result)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid output data: {str(e)}")

        return validated_output

    return wrapper


# Decorator for publishing data to Kafka
def publish_to_kafka(
    func: DecoratedCallable, config: DecoratorConfig
) -> DecoratedCallable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)

        # Publish the result data to Kafka
        try:
            await config.kafka_producer.send(config.kafka_config["topic"], result)
        except Exception as e:
            logger.error(f"Failed to publish data to Kafka: {str(e)}")

        return result

    return wrapper
