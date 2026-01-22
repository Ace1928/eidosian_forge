import asyncio
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple

# Ensuring the use of aiofiles for asynchronous file operations
import aiofiles

# Importing custom async file handler for enhanced logging capabilities
import async_file_handler as afh


class AsyncCache:
    def __init__(
        self,
        cache_lifetime: int = 60,
        cache_maxsize: int = 128,
        enable_caching: bool = True,
        file_cache_path: str = "file_cache.pkl",
        call_threshold: int = 10,
        cache_key_strategy: Optional[Callable] = None,
    ):
        """
        Initializes the AsyncCache instance with meticulously defined parameters, crafting a highly adaptive, efficient, and robust caching mechanism. This constructor method is designed to cater to diverse caching requirements with precision, ensuring optimal performance and flexibility.

        Args:
            cache_lifetime (int): Specifies the duration (in seconds) for which cache entries remain valid, facilitating precise control over cache freshness. Defaults to 60 seconds.
            cache_maxsize (int): Defines the maximum number of entries the in-memory cache can hold, enabling efficient memory usage management. Defaults to 128 entries.
            enable_caching (bool): A boolean flag to toggle caching functionality according to application needs. Defaults to True, enabling caching.
            file_cache_path (str): Designates the file path for persistent cache storage, ensuring data durability across application restarts. Defaults to "file_cache.pkl".
            call_threshold (int): Determines the number of accesses after which cache entries are persisted to disk, optimizing disk I/O operations. Defaults to 10 accesses.
            cache_key_strategy (Optional[Callable]): Allows for a custom cache key generation strategy, providing flexibility in defining cache key logic. Defaults to None, using the default strategy.

        Attributes:
            cache (Dict[Tuple[Any, ...], Any]): An in-memory cache dictionary to store cache entries.
            file_cache (Dict[Tuple[Any, ...], Any]): A dictionary to manage cache entries persisted to disk.
            cache_lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations on the in-memory cache.
            file_io_lock (asyncio.Lock): An asyncio lock to manage thread-safe file I/O operations for the file cache.
            call_counter (Dict[Tuple[Any, ...], int]): A dictionary to track access counts for cache entries, aiding in persistence logic.
            logger (logging.Logger): A logger configured for detailed logging of cache operations.
            loop (asyncio.AbstractEventLoop): The event loop in which cache maintenance tasks are scheduled.
        """
        self.cache_lifetime = cache_lifetime
        self.cache_maxsize = cache_maxsize
        self.enable_caching = enable_caching
        self.file_cache_path = file_cache_path
        self.call_threshold = call_threshold
        self.cache_key_strategy = cache_key_strategy or self._default_cache_key_strategy
        self.cache = {}
        self.file_cache = {}
        self.cache_lock = asyncio.Lock()
        self.file_io_lock = asyncio.Lock()
        self.call_counter = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(afh.AsyncFileHandler("cache.log"))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._initialize_cache())
        self.loop.create_task(self.maintain_cache())

    def _default_cache_key_strategy(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, ...]:
        """
        A default strategy for generating cache keys based on function arguments. This method ensures that each cache entry can be uniquely identified, facilitating precise and efficient cache retrieval.

        Args:
            func (Callable): The function for which the cache key is being generated.
            *args: Positional arguments of the function call.
            **kwargs: Keyword arguments of the function call.

        Returns:
            Tuple[Any, ...]: A tuple representing the unique cache key.
        """
        return self.generate_cache_key(func, *args, **kwargs)

    async def _initialize_cache(self):
        """
        Asynchronously initializes the cache configurations, meticulously setting up both in-memory and file-based caches. This method ensures readiness for operational use without impeding the main event loop's execution flow.

        The initialization process encompasses the retrieval of existing cache entries from the file cache, if available, and their subsequent loading into the in-memory cache. This approach guarantees the persistence of cache data across application restarts, thereby enhancing data retrieval efficiency and application performance.

        Raises:
            IOError: If there's an error reading the file cache from the specified path.
            pickle.UnpicklingError: If an error occurs during the deserialization of the file cache.
        """
        if self.enable_caching:
            self.cache.clear()  # Explicitly clear the in-memory cache to ensure a fresh start.
            if os.path.exists(self.file_cache_path):
                async with self.file_io_lock:
                    try:
                        async with aiofiles.open(
                            self.file_cache_path, mode="rb"
                        ) as file:
                            file_contents = await file.read()
                            self.file_cache = pickle.loads(file_contents)
                    except (IOError, pickle.UnpicklingError) as e:
                        self.logger.error(f"Failed to initialize file-based cache: {e}")
                        raise
            self.logger.debug(
                "Cache initialization process completed asynchronously, ensuring high performance and responsiveness."
            )

    async def background_cache_persistence(self, key: Tuple[Any, ...], value: Any):
        """
        Asynchronously schedules a task to persist cache updates to the file cache. This method is designed to facilitate non-blocking operations and maintain thread safety within an asyncio context.

        Args:
            key (Tuple[Any, ...]): The cache key under which the value is to be stored.
            value (Any): The value to be stored in the cache.

        Logs:
            An error message if the cache persistence task encounters an exception.
        """
        try:
            await asyncio.create_task(self._save_to_file_cache(key, value))
        except Exception as e:
            self.logger.error(
                f"Error initiating background cache persistence for key {key}: {e}"
            )

    def generate_cache_key(
        self, func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """
        Constructs a unique cache key derived from the function name, its arguments, and keyword arguments. This method employs serialization of arguments and keyword arguments to ensure the uniqueness of each cache key.

        Args:
            func (Callable): The function for which the cache key is being generated.
            args (Tuple[Any, ...]): The positional arguments of the function.
            kwargs (Dict[str, Any]): The keyword arguments of the function.

        Returns:
            Tuple[Any, ...]: A tuple representing the unique cache key, ensuring precise cache retrieval.

        Logs:
            A debug message indicating the successful generation of a cache key.
        """
        sorted_kwargs = tuple(
            sorted(kwargs.items(), key=lambda item: item[0])
        )  # Ensure deterministic ordering.
        serialized_args = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
        serialized_kwargs = pickle.dumps(
            sorted_kwargs, protocol=pickle.HIGHEST_PROTOCOL
        )
        cache_key = (func.__name__, serialized_args, serialized_kwargs)
        self.logger.debug(
            f"Generated cache key: {cache_key} for function {func.__name__}"
        )
        return cache_key

    async def cache_logic(
        self, key: Tuple[Any, ...], func: Callable, *args, **kwargs
    ) -> Any:
        """
        Implements the caching logic, checking for cache hits in both in-memory and file caches.
        If a cache miss occurs, the function is executed and the result is cached.
        This method handles both synchronous and asynchronous functions dynamically, ensuring thread safety
        and non-blocking operations within an asyncio context.

        Args:
            key (Tuple[Any, ...]): The cache key under which the result is to be stored.
            func (Callable): The function to be executed in case of a cache miss.
            *args: Variable length argument list for the function `func`.
            **kwargs: Arbitrary keyword arguments for the function `func`.

        Returns:
            Any: The result from the cache or the executed function.

        Raises:
            Exception: Propagates exceptions from the function execution or cache operations, logging the error details.

        Logs:
            Debug messages indicating cache hits, misses, and the eviction of least recently used items.
            Error messages capturing exceptions encountered during cache operations.
        """
        async with self.cache_lock:
            try:
                # Attempt to retrieve the result from the in-memory cache
                if key in self.cache:
                    self.cache.move_to_end(key)  # Mark as recently used
                    self.call_counter[key] += 1  # Increment call counter
                    self.logger.debug(f"Cache hit for {key}.")
                    return self.cache[key][0]  # Return the cached result

                # Attempt to retrieve the result from the file cache
                elif key in self.file_cache:
                    result = self.file_cache[key]
                    self.logger.debug(f"File cache hit for {key}.")

                # Handle cache miss
                else:
                    # Execute the function asynchronously if it's a coroutine
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    # Execute the function in a separate thread if it's synchronous
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)

                    self.logger.debug(
                        f"Cache miss for {key}. Function executed and result cached."
                    )

                    # Update the in-memory cache with the new result
                    self.cache[key] = (
                        result,
                        datetime.utcnow() + timedelta(seconds=self.cache_lifetime),
                    )
                    self.call_counter[key] = (
                        1  # Initialize call counter for the new key
                    )

                    # Evict the least recently used item if the cache exceeds its maximum size
                    if len(self.cache) > self.cache_maxsize:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                        del self.call_counter[oldest_key]
                        self.logger.debug(
                            f"Evicted least recently used cache item: {oldest_key}"
                        )

                    # Persist cache updates to the file cache asynchronously
                    await self.background_cache_persistence(key, self.cache[key])
                    self.logger.debug(f"Result cached for key: {key}")

                return result

            except Exception as e:
                self.logger.error(f"Error in cache logic for key {key}: {e}")
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
                async with self.file_io_lock:
                    async with aiofiles.open(self.file_cache_path, mode="rb") as file:
                        file_content = await file.read()
                        # Utilizing asyncio.to_thread to offload the blocking operation, pickle.loads, to a separate thread.
                        self.file_cache = await asyncio.to_thread(
                            pickle.loads, file_content
                        )
                        self.logger.debug(
                            f"File cache successfully loaded from {self.file_cache_path}."
                        )
            else:
                # Initializing an empty cache if the file cache does not exist.
                self.file_cache = {}
                self.logger.debug("File cache not found. Initialized an empty cache.")
        except Exception as error:
            # Logging the exception details in case of failure to load the file cache.
            self.logger.error(f"Failed to load file cache due to error: {error}")
            raise

    async def _dump_file_cache(self) -> None:
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
            # Serialize the file cache first to avoid blocking in async context
            serialized_cache = await asyncio.to_thread(
                pickle.dumps, self.file_cache, pickle.HIGHEST_PROTOCOL
            )
            async with self.file_io_lock:
                async with aiofiles.open(self.file_cache_path, mode="wb") as file:
                    # Asynchronously write the serialized cache to the file
                    await file.write(serialized_cache)
                    self.logger.debug("File cache has been successfully saved to disk.")
        except Exception as error:
            # Logging the exception details in case of failure to save the file cache.
            self.logger.error(f"Failed to save file cache due to error: {error}")
            raise

    async def clean_expired_cache_entries(self):
        """
        Asynchronously cleans expired entries from both in-memory and file caches.
        This method meticulously iterates over the cache entries, identifying and removing those that have surpassed their expiration time.
        It ensures thread safety through the use of an asyncio.Lock, thereby preventing race conditions during the cleanup process.
        Additionally, it logs detailed information about the cleanup operation, aiding in debugging and monitoring of cache behavior.

        Raises:
            Exception: Logs a detailed error message indicating the failure to clean expired cache entries due to an encountered exception.
        """
        async with self.cache_lock:
            try:
                current_time = datetime.utcnow()
                # Cleaning in-memory cache
                expired_keys = [
                    key
                    for key, (_, expiration_time) in self.cache.items()
                    if expiration_time < current_time
                ]
                for key in expired_keys:
                    del self.cache[key]
                    self.logger.debug(f"Removed expired cache entry: {key}")

                # Cleaning file cache
                expired_keys_file_cache = [
                    key
                    for key, (_, expiration_time) in self.file_cache.items()
                    if expiration_time < current_time
                ]
                for key in expired_keys_file_cache:
                    del self.file_cache[key]
                if expired_keys_file_cache:
                    await self._dump_file_cache()
                self.logger.debug(
                    f"Expired entries removed from file cache: {expired_keys_file_cache}"
                )
            except Exception as e:
                self.logger.error(f"Error cleaning expired cache entries: {e}")
                raise

    async def _evict_lru_from_memory(self):
        """
        Asynchronously evicts the least recently used (LRU) item from the in-memory cache.
        This method employs an asyncio.Lock to ensure thread safety, preventing race conditions during the eviction process.
        It logs detailed information about the eviction, including the key of the evicted item, to facilitate debugging and monitoring of cache behavior.

        Raises:
            Exception: Logs a detailed error message indicating the failure to evict the LRU item due to an encountered exception.
        """
        async with self.cache_lock:
            try:
                if len(self.cache) > self.cache_maxsize:
                    # Identify the least recently used item
                    lru_key = next(iter(self.cache))
                    # Evict the identified LRU item
                    del self.cache[lru_key]
                    del self.call_counter[lru_key]
                    self.logger.debug(
                        f"Evicted least recently used cache item: {lru_key}"
                    )

                    # Asynchronously save the updated cache state to the file cache
                    await self._dump_file_cache()

                    self.logger.info(
                        f"Successfully evicted LRU item from memory: {lru_key} and updated file cache."
                    )
                else:
                    self.logger.debug("No eviction needed. Cache size within limits.")
            except Exception as e:
                self.logger.error(f"Error evicting LRU item from memory: {e}")
                raise

    async def maintain_cache(self):
        """
        Periodically checks and maintains the cache by cleaning expired entries and evicting LRU items.
        This method runs in a separate asynchronous task, ensuring efficient cache management without blocking the main execution flow.
        It incorporates advanced logic to monitor the cache's state and initiates a graceful shutdown if the cache remains unchanged for an extended period, indicating potential inactivity or improper closure.

        Raises:
            Exception: Logs a detailed error message indicating the failure to maintain the cache due to an encountered exception.
        """
        try:
            unchanged_cycles = (
                0  # Counter for the number of cycles with no cache changes
            )
            is_main_program = (
                __name__ == "__main__"
            )  # Check if running as the main program
            sleep_duration = (
                60 if is_main_program else 10
            )  # Sleep duration based on execution context

            while True:
                await asyncio.sleep(
                    sleep_duration
                )  # Pause execution for the determined duration

                # Capture the state of caches before maintenance to detect changes
                initial_memory_cache_state = str(self.cache)
                initial_file_cache_state = str(self.file_cache)

                # Execute cache maintenance operations
                await self.clean_expired_cache_entries()  # Clean expired entries from caches
                await self._evict_lru_from_memory()  # Evict least recently used items from memory cache

                # Determine if the cache states have changed following maintenance operations
                memory_cache_changed = initial_memory_cache_state != str(self.cache)
                file_cache_changed = initial_file_cache_state != str(self.file_cache)

                # Log the completion of cache maintenance based on the program's execution context
                context_message = (
                    "the main program"
                    if is_main_program
                    else "the imported module context"
                )
                self.logger.debug(f"Cache maintenance completed in {context_message}.")

                # Update the unchanged cycles counter based on cache state changes
                if memory_cache_changed or file_cache_changed:
                    unchanged_cycles = 0  # Reset counter if changes were detected
                else:
                    unchanged_cycles += (
                        1  # Increment counter if no changes were detected
                    )

                # Initiate a graceful shutdown if caches have remained unchanged for 10 cycles
                if unchanged_cycles >= 10:
                    self.logger.info(
                        "No cache changes detected for 10 cycles. Initiating graceful shutdown."
                    )
                    # Placeholder for graceful shutdown logic
                    # This should include tasks such as closing database connections, stopping async tasks,
                    # and any other cleanup required before safely terminating the program.
                pass

        except Exception as e:
            self.logger.error(f"Error in cache maintenance task: {e}")

    async def attempt_cache_retrieval(self, key: Tuple[Any, ...]) -> Optional[Any]:
        """
        Attempts to retrieve the cached result for the given key from both in-memory and file caches. This method employs a meticulous and systematic approach to ensure the highest likelihood of cache hit, thereby minimizing data retrieval latency and enhancing application performance. It leverages asynchronous programming paradigms to maintain non-blocking operations, ensuring the responsiveness of the application.

        Args:
            key (Tuple[Any, ...]): The key for which the cache retrieval is attempted.

        Returns:
            Optional[Any]: The cached result if found; otherwise, None.

        Raises:
            Exception: Logs and re-raises any exceptions encountered during the cache retrieval process to ensure robust error handling and maintain application stability.
        """
        try:
            async with self.cache_lock:
                # Check for the key in the in-memory cache
                if key in self.cache:
                    self.logger.debug(f"In-memory cache hit for {key}.")
                    return self.cache[key]
                # Check for the key in the file cache
                elif key in self.file_cache:
                    self.logger.debug(f"File cache hit for {key}.")
                    result = self.file_cache[key]
                    # Evict the least recently used item from memory if the cache is at its max size
                    if len(self.cache) >= self.cache_maxsize:
                        await self._evict_lru_from_memory()
                    # Update the in-memory cache with the result from the file cache
                    self.cache[key] = result
                    return result
            return None
        except Exception as e:
            self.logger.error(f"Error attempting cache retrieval for key {key}: {e}")
            raise

    async def update_cache(
        self, key: Tuple[Any, ...], result: Any, is_async: bool = True
    ):
        """
        Updates the cache with the given key and result. This method manages cache sizes, evictions, and time-based expiration with unparalleled precision and efficiency. It ensures the cache remains optimally utilized and up-to-date, leveraging asynchronous programming paradigms to maintain non-blocking operations and ensure the responsiveness of the application.

        Args:
            key (Tuple[Any, ...]): The key under which the result is to be stored in the cache.
            result (Any): The result to be stored in the cache.
            is_async (bool): A flag indicating whether the cache update should be performed asynchronously.

        Raises:
            Exception: Logs and re-raises any exceptions encountered during the cache update process to ensure robust error handling and maintain application stability.
        """
        try:
            current_time = datetime.utcnow()
            expiration_time = current_time + timedelta(seconds=self.cache_lifetime)
            async with self.cache_lock:
                # Evict the least recently used item from memory if the cache is at its max size
                if len(self.cache) >= self.cache_maxsize:
                    await self._evict_lru_from_memory()
                # Update the in-memory cache with the new result and expiration time
                self.cache[key] = (result, expiration_time)
                # Check if the call counter for the key has reached the threshold for file cache persistence
                if self.call_counter.get(key, 0) >= self.call_threshold:
                    self.call_counter[key] = 0
                    if is_async:
                        # Save to file cache asynchronously if the is_async flag is True
                        await self._save_to_file_cache(key, (result, expiration_time))
                else:
                    # Increment the call counter and persist the cache update in the background
                    self.call_counter[key] = self.call_counter.get(key, 0) + 1
                    await self.background_cache_persistence(
                        key, (result, expiration_time)
                    )
                    self.logger.debug(
                        f"Cache updated for key: {key}. Cache size: {len(self.cache)}"
                    )
        except Exception as e:
            self.logger.error(f"Error updating cache for key {key}: {e}")
            raise

    async def _save_to_file_cache(
        self, key: Tuple[Any, ...], value: Tuple[Any, datetime]
    ):
        """
        Asynchronously saves a key-value pair to the file cache. This method leverages advanced asynchronous file operations to ensure the event loop remains unblocked, thereby maintaining high application responsiveness while ensuring data persistence. It encapsulates the file cache update operation within a try-except block to gracefully handle potential exceptions, ensuring the application's robustness and reliability.

        Args:
            key (Tuple[Any, ...]): The key under which the value is to be stored in the file cache.
            value (Tuple[Any, datetime]): The value and its expiration time to be stored in the file cache.

        Raises:
            Exception: Logs and re-raises any exceptions encountered during the file cache update process to ensure robust error handling and maintain application stability.
        """
        try:
            async with self.cache_lock:
                # Update the file cache with the new key-value pair
                self.file_cache[key] = value
                async with self.file_io_lock:
                    # Open the file cache path asynchronously for writing
                    async with aiofiles.open(self.file_cache_path, "wb") as f:
                        await f.write(
                            pickle.dumps(self.file_cache, pickle.HIGHEST_PROTOCOL)
                        )
            logging.debug(
                f"File cache asynchronously updated for key: {key} with the highest level of efficiency and reliability."
            )
        except Exception as e:
            logging.error(
                f"Error saving to file cache asynchronously for key {key}: {e}"
            )
