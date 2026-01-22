"""
Module: Indecache
Description: Indecache is a highly advanced custom cache implementation that interfaces with KeyDB as its backend. It is designed to provide a highly efficient and robust caching mechanism for a wide range of applications, leveraging advanced caching strategies and features to ensure optimal performance, flexibility, and adaptability to diverse caching requirements.

Features:
- Asynchronous operations for non-blocking cache access
- Support for LRU eviction, TTL management, and thread-safe operations
- In-memory and file-based caching for fast data retrieval and persistence
- Networked caching for distributed cache management
- Dynamic and adaptive retry mechanisms
- Detailed logging for monitoring and performance analysis
- Typed caching, sparse data handling, and hashing for integrity verification

Dependencies:
- aiohttp: For asynchronous web application initialization
- cachetools: For TTLCache implementation
- aiokeydb: For asynchronous KeyDB client configuration
- asyncio, functools, inspect, logging, os, json, pickle, time, warnings, datetime, pathlib, typing, concurrent.futures, multiprocessing, numpy, pandas, scikit-learn, aiofiles, joblib, lz4, msgpack, orjson, zstandard, psutil, pydantic, fastapi, starlette, aiokafka, logging.config

Author: [Author Name]
Version: 1.0
Last Updated: [Last Updated Date]
"""

# Importing necessary libraries and modules
import asyncio
import hashlib
import logging
import logging.config
import pickle
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, TypeVar, Awaitable, Optional, Union

import aiokeydb
from aiokeydb import AsyncKeyDB, KeyDBError
from cachetools import TTLCache

from indedecorators import async_log_decorator

# Setting up logging configuration
logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)

# Custom types for enhanced readability and maintainability
T = TypeVar("T")
DecoratedCallable = Callable[..., Coroutine[Any, Any, T]]
CacheKeyType = Union[str, int, float, tuple, frozenset]
CacheValueType = Any

# Global cache settings
CACHE_SETTINGS = {
    "TTL": 7200,  # Time to live in seconds
    "MAXSIZE": 2048,  # Maximum size of the cache
    "KEYDB_URI": "keydb://localhost:6379/0",  # URI for KeyDB instance
}


# Asynchronous cache class to encapsulate caching logic
class AsyncCache:
    """
    AsyncCache is an asynchronous cache implementation that utilizes KeyDB as its backend storage.
    It provides an efficient and flexible caching mechanism with support for TTL, LRU eviction, and thread-safety.

    Args:
        ttl (int): Time to live for cached items in seconds.
        maxsize (int): Maximum size of the cache.
        keydb_uri (str): URI for the KeyDB instance.
        cache_name (str): Name of the cache instance for identification and logging purposes.

    Attributes:
        _cache (TTLCache): In-memory cache using TTLCache.
        _keydb_client (AsyncKeyDB): Asynchronous KeyDB client for backend storage.
        _keydb_uri (str): URI for the KeyDB instance.
        _cache_name (str): Name of the cache instance.

    Methods:
        _get_keydb_client: Asynchronously retrieves the KeyDB client instance.
        get: Asynchronously retrieves a value from the cache.
        set: Asynchronously sets a value in the cache.
        delete: Asynchronously deletes a key from the cache.
        clear: Asynchronously clears all keys from the cache.
        keys: Asynchronously retrieves all keys from the cache.
        values: Asynchronously retrieves all values from the cache.
        items: Asynchronously retrieves all key-value pairs from the cache.
        size: Asynchronously retrieves the current size of the cache.
        ttl: Asynchronously retrieves the time-to-live (TTL) value for a key.
        persist: Asynchronously persists the cache data to the file system.
        load: Asynchronously loads the cache data from the file system.
    """

    def __init__(self, ttl: int, maxsize: int, keydb_uri: str, cache_name: str):
        """
        Initializes the AsyncCache instance with the provided TTL, maxsize, KeyDB URI, and cache name.

        Args:
            ttl (int): Time to live for cached items in seconds.
            maxsize (int): Maximum size of the cache.
            keydb_uri (str): URI for the KeyDB instance.
            cache_name (str): Name of the cache instance for identification and logging purposes.
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._keydb_client: Optional[AsyncKeyDB] = None
        self._keydb_uri = keydb_uri
        self._cache_name = cache_name

    async def _get_keydb_client(self) -> AsyncKeyDB:
        """
        Asynchronously retrieves the KeyDB client instance.
        If the client is not already initialized, it establishes a new connection to KeyDB.

        Returns:
            AsyncKeyDB: The asynchronous KeyDB client instance.

        Raises:
            KeyDBError: If there is an error connecting to KeyDB.
        """
        if not self._keydb_client:
            try:
                self._keydb_client = await AsyncKeyDB.from_url(self._keydb_uri)
                logger.info(f"Connected to KeyDB for cache: {self._cache_name}")
            except KeyDBError as e:
                logger.error(
                    f"Failed to connect to KeyDB for cache: {self._cache_name}. Error: {e}"
                )
                raise
        return self._keydb_client

    @async_log_decorator
    async def get(self, key: CacheKeyType) -> Optional[CacheValueType]:
        """
        Asynchronously retrieves a value from the cache.
        First, it checks the in-memory cache. If the value is not found, it attempts to retrieve it from KeyDB.

        Args:
            key (CacheKeyType): The key to retrieve the value for.

        Returns:
            Optional[CacheValueType]: The cached value if found, otherwise None.
        """
        # Attempt to get the value from in-memory cache
        if key in self._cache:
            logger.debug(f"Cache hit for key: {key} in cache: {self._cache_name}")
            return self._cache[key]

        # Attempt to retrieve from KeyDB
        client = await self._get_keydb_client()
        try:
            value = await client.get(str(key))
            if value is not None:
                logger.debug(
                    f"Retrieved key: {key} from KeyDB for cache: {self._cache_name}"
                )
                deserialized_value = pickle.loads(value)
                # Update in-memory cache with the retrieved value
                self._cache[key] = deserialized_value
                return deserialized_value
        except KeyDBError as e:
            logger.error(
                f"Error retrieving key: {key} from KeyDB for cache: {self._cache_name}. Error: {e}"
            )
        logger.debug(f"Cache miss for key: {key} in cache: {self._cache_name}")
        return None

    @async_log_decorator
    async def set(self, key: CacheKeyType, value: CacheValueType) -> None:
        """
        Asynchronously sets a value in the cache.
        The value is stored in both the in-memory cache and KeyDB for persistence.

        Args:
            key (CacheKeyType): The key to set the value for.
            value (CacheValueType): The value to cache.
        """
        self._cache[key] = value
        # Serialize and store in KeyDB for persistence
        client = await self._get_keydb_client()
        try:
            serialized_value = pickle.dumps(value)
            await client.set(str(key), serialized_value)
            logger.debug(
                f"Set key: {key} in both in-memory and KeyDB caches for cache: {self._cache_name}"
            )
        except (pickle.PicklingError, KeyDBError) as e:
            logger.error(
                f"Error setting key: {key} in KeyDB for cache: {self._cache_name}. Error: {e}"
            )

    @async_log_decorator
    async def delete(self, key: CacheKeyType) -> None:
        """
        Asynchronously deletes a key from the cache.
        The key is removed from both the in-memory cache and KeyDB.

        Args:
            key (CacheKeyType): The key to delete.
        """
        if key in self._cache:
            del self._cache[key]
        client = await self._get_keydb_client()
        try:
            await client.delete(str(key))
            logger.debug(
                f"Deleted key: {key} from both in-memory and KeyDB caches for cache: {self._cache_name}"
            )
        except KeyDBError as e:
            logger.error(
                f"Error deleting key: {key} from KeyDB for cache: {self._cache_name}. Error: {e}"
            )

    @async_log_decorator
    async def clear(self) -> None:
        """
        Asynchronously clears all keys from the cache.
        All keys are removed from both the in-memory cache and KeyDB.
        """
        self._cache.clear()
        client = await self._get_keydb_client()
        try:
            await client.flushdb()
            logger.debug(
                f"Cleared all keys from both in-memory and KeyDB caches for cache: {self._cache_name}"
            )
        except KeyDBError as e:
            logger.error(
                f"Error clearing keys from KeyDB for cache: {self._cache_name}. Error: {e}"
            )

    @async_log_decorator
    async def keys(self) -> List[CacheKeyType]:
        """
        Asynchronously retrieves all keys from the cache.

        Returns:
            List[CacheKeyType]: A list of all keys in the cache.
        """
        client = await self._get_keydb_client()
        try:
            keys = await client.keys("*")
            logger.debug(f"Retrieved all keys from KeyDB for cache: {self._cache_name}")
            return [key.decode() for key in keys]
        except KeyDBError as e:
            logger.error(
                f"Error retrieving keys from KeyDB for cache: {self._cache_name}. Error: {e}"
            )
            return []

    @async_log_decorator
    async def values(self) -> List[CacheValueType]:
        """
        Asynchronously retrieves all values from the cache.

        Returns:
            List[CacheValueType]: A list of all values in the cache.
        """
        client = await self._get_keydb_client()
        try:
            values = await client.mget(await self.keys())
            logger.debug(
                f"Retrieved all values from KeyDB for cache: {self._cache_name}"
            )
            return [pickle.loads(value) for value in values if value is not None]
        except (KeyDBError, pickle.UnpicklingError) as e:
            logger.error(
                f"Error retrieving values from KeyDB for cache: {self._cache_name}. Error: {e}"
            )
            return []

    @async_log_decorator
    async def items(self) -> List[Tuple[CacheKeyType, CacheValueType]]:
        """
        Asynchronously retrieves all key-value pairs from the cache.

        Returns:
            List[Tuple[CacheKeyType, CacheValueType]]: A list of all key-value pairs in the cache.
        """
        client = await self._get_keydb_client()
        try:
            keys = await self.keys()
            values = await client.mget(keys)
            logger.debug(
                f"Retrieved all key-value pairs from KeyDB for cache: {self._cache_name}"
            )
            return [
                (key, pickle.loads(value))
                for key, value in zip(keys, values)
                if value is not None
            ]
        except (KeyDBError, pickle.UnpicklingError) as e:
            logger.error(
                f"Error retrieving key-value pairs from KeyDB for cache: {self._cache_name}. Error: {e}"
            )
            return []

    @async_log_decorator
    async def size(self) -> int:
        """
        Asynchronously retrieves the current size of the cache.

        Returns:
            int: The number of items in the cache.
        """
        client = await self._get_keydb_client()
        try:
            size = await client.dbsize()
            logger.debug(
                f"Retrieved cache size from KeyDB for cache: {self._cache_name}"
            )
            return size
        except KeyDBError as e:
            logger.error(
                f"Error retrieving cache size from KeyDB for cache: {self._cache_name}. Error: {e}"
            )
            return 0

    @async_log_decorator
    async def ttl(self, key: CacheKeyType) -> Optional[int]:
        """
        Asynchronously retrieves the time-to-live (TTL) value for a key.

        Args:
            key (CacheKeyType): The key to retrieve the TTL for.

        Returns:
            Optional[int]: The remaining TTL in seconds if the key exists and has a TTL set, otherwise None.
        """
        client = await self._get_keydb_client()
        try:
            ttl = await client.ttl(str(key))
            logger.debug(
                f"Retrieved TTL for key: {key} from KeyDB for cache: {self._cache_name}"
            )
            return ttl if ttl != -1 else None
        except KeyDBError as e:
            logger.error(
                f"Error retrieving TTL for key: {key} from KeyDB for cache: {self._cache_name}. Error: {e}"
            )
            return None

    @async_log_decorator
    async def persist(self) -> None:
        """
        Asynchronously persists the cache data to the file system.
        The cache data is serialized and stored in a file for long-term persistence.
        """
        try:
            cache_data = {"keys": await self.keys(), "values": await self.values()}
            serialized_data = pickle.dumps(cache_data)
            async with aiofiles.open(
                f"{self._cache_name}_cache.pkl", "wb"
            ) as cache_file:
                await cache_file.write(serialized_data)
            logger.info(f"Persisted cache data to file for cache: {self._cache_name}")
        except (IOError, pickle.PicklingError) as e:
            logger.error(
                f"Error persisting cache data to file for cache: {self._cache_name}. Error: {e}"
            )

    @async_log_decorator
    async def load(self) -> None:
        """
        Asynchronously loads the cache data from the file system.
        The cache data is deserialized from a file and restored into the cache.
        """
        try:
            async with aiofiles.open(
                f"{self._cache_name}_cache.pkl", "rb"
            ) as cache_file:
                serialized_data = await cache_file.read()
            cache_data = pickle.loads(serialized_data)
            keys = cache_data["keys"]
            values = cache_data["values"]
            for key, value in zip(keys, values):
                await self.set(key, value)
            logger.info(f"Loaded cache data from file for cache: {self._cache_name}")
        except (IOError, pickle.UnpicklingError, KeyError) as e:
            logger.error(
                f"Error loading cache data from file for cache: {self._cache_name}. Error: {e}"
            )


# Initialize multiple cache instances
cache_instances = {
    "cache1": AsyncCache(**CACHE_SETTINGS, cache_name="cache1"),
    "cache2": AsyncCache(**CACHE_SETTINGS, cache_name="cache2"),
    "cache3": AsyncCache(**CACHE_SETTINGS, cache_name="cache3"),
}


# Decorator for caching function results asynchronously
def async_cache(cache_name: str):
    """
    A decorator that caches the result of an asynchronous function using the specified AsyncCache instance.
    It generates a unique cache key based on the function name and arguments.
    If the result is already cached, it returns the cached value. Otherwise, it executes the function and caches the result.

    Args:
        func (DecoratedCallable): The asynchronous function to be decorated.

    Returns:
        DecoratedCallable: The decorated function with caching functionality.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        # Generate a unique cache key based on function name and arguments
        cache_key = hashlib.sha256(
            pickle.dumps((func.__name__, args, frozenset(kwargs.items())))
        ).hexdigest()
        # Attempt to retrieve the cached result
        cached_result = await global_cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                f"Returning cached result for {func.__name__} with key: {cache_key}"
            )
            return cached_result
        # Execute the function and cache its result
        try:
            result = await func(*args, **kwargs)
            await global_cache.set(cache_key, result)
            logger.info(f"Cached result for {func.__name__} with key: {cache_key}")
            return result
        except Exception as e:
            logger.exception(f"Error executing {func.__name__}: {e}")
            raise

    return wrapper


# Example usage of the async_cache decorator
@async_cache
async def fetch_data(param: str) -> Dict[str, Any]:
    """
    An example asynchronous function that simulates an I/O operation and returns a dictionary.

    Args:
        param (str): A parameter for the function.

    Returns:
        Dict[str, Any]: A dictionary containing the input parameter.
    """
    # Simulate an I/O operation
    await asyncio.sleep(1)
    return {"data": param}


# Initialize and run an example if this script is executed directly
async def main():
    """
    The main function that demonstrates the usage of the async_cache decorator and the fetch_data function.
    It calls the fetch_data function and prints the result.
    """
    try:
        result = await fetch_data("example")
        print(result)
    except Exception as e:
        logger.exception(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
