class StandardDecorator:
"""
Refactored from /home/lloyd/EVIE/standard_decorator.py
For utilisation of the refined, aligned, standardised, enhanced and improved
methods and utilities found in the files:


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

        self.logger = self.setup_logging()

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

        self.validation_lock = asyncio.Lock()  # Lock for argument validation

        self.logging_lock = asyncio.Lock()  # Lock for logging operations

        self.file_io_lock = asyncio.Lock()  # Lock for file I/O operations

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

        self.cache_key_strategy = cache_key_strategy or self._default_cache_key_strategy

default_cache_key_strategy = async_cache.default_cache_key_strategy()
_run_async_coroutine = sync_async_launcher._run_async_coroutine()



    def _default_cache_key_strategy(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, ...]:
        """
        Default cache key strategy that generates a cache key based on the function arguments.
        """
        return self.generate_cache_key(func, *args, **kwargs)

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
            async with self.file_io_lock:
                async with aiofiles.open(self.file_cache_path, "wb") as f:
                    await f.write(pickle.dumps(self.cache))
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
        loop = asyncio.get_running_loop()
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
        async with self.logging_lock:
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
        performance metrics
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
                async with self.logging_lock:
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
        loop = asyncio.get_running_loop()
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

    async def _initialize_cache(self):
        """
        Initializes caching mechanisms based on configuration asynchronously.
        """
        if self.enable_caching:
            # Initialize in-memory cache
            self.cache = {}
            # Check and initialize file-based cache if specified
            if os.path.exists(self.file_cache_path):
                async with self.file_io_lock:
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
                async with self.file_io_lock:
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
            # Serialize the file cache first to avoid blocking in async context
            serialized_cache = await asyncio.to_thread(
                pickle.dumps, self.file_cache, pickle.HIGHEST_PROTOCOL
            )
            async with self.file_io_lock:
                async with aiofiles.open(self.file_cache_path, mode="wb") as file:
                    # Asynchronously write the serialized cache to the file
                    await file.write(serialized_cache)
                    logging.debug("File cache has been successfully saved to disk.")
        except Exception as error:
            # Logging the exception details in case of failure to save the file cache.
            logging.error(f"Failed to save file cache due to error: {error}")
            raise

    async def clean_expired_cache_entries(self):
        """
        Cleans expired entries from both in-memory and file caches asynchronously.
        This method iterates over the cache entries and removes those that have expired,
        ensuring thread safety by utilizing an asyncio.Lock.
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
                    logging.debug(f"Removed expired cache entry: {key}")

                # Cleaning file cache
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
                raise

    async def _evict_lru_from_memory(self):
        """
        Asynchronously evicts the least recently used (LRU) item from the in-memory cache.
        This method ensures thread safety by utilizing an asyncio.Lock, preventing race conditions
        during the eviction process. It logs detailed information about the eviction process,
        including the key of the evicted item, to aid in debugging and monitoring of cache behavior.

        Raises:
            Exception: Logs an error message indicating the failure to evict the LRU item due to an encountered exception.
        """
        async with self.cache_lock:
            try:
                if len(self.cache) > self.cache_maxsize:
                    # Identify the least recently used item
                    lru_key = next(iter(self.cache))
                    # Evict the identified LRU item
                    del self.cache[lru_key]
                    del self.call_counter[lru_key]
                    logging.debug(f"Evicted least recently used cache item: {lru_key}")

                    # Asynchronously save the updated cache state to the file cache
                    await self._save_file_cache()

                    logging.info(
                        f"Successfully evicted LRU item from memory: {lru_key} and updated file cache."
                    )
                else:
                    logging.debug("No eviction needed. Cache size within limits.")
            except Exception as e:
                logging.error(f"Error evicting LRU item from memory: {e}")
                raise

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
            async with self.cache_lock:
                if key in self.cache:
                    logging.debug(f"In-memory cache hit for {key}")
                    return self.cache[key]
                elif key in self.file_cache:
                    logging.debug(f"File cache hit for {key}")
                    result = self.file_cache[key]
                    if len(self.cache) >= self.cache_maxsize:
                        await self._evict_lru_from_memory()
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
            async with self.cache_lock:
                if len(self.cache) >= self.cache_maxsize:
                    await self._evict_lru_from_memory()
                self.cache[key] = (result, expiration_time)
                if self.call_counter.get(key, 0) >= self.call_threshold:
                    self.call_counter[key] = 0
                    if is_async:
                        await self._save_to_file_cache(key, (result, expiration_time))
                else:
                    await self.background_cache_persistence(
                        key, (result, expiration_time)
                    )
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
            async with self.cache_lock:
                self.file_cache[key] = value
                async with self.file_io_lock:
                    async with aiofiles.open(self.file_cache_path, "wb") as f:
                        await f.write(
                            pickle.dumps(self.file_cache, pickle.HIGHEST_PROTOCOL)
                        )
            logging.debug(f"File cache asynchronously updated for key: {key}")
        except Exception as e:
            logging.error(
                f"Error saving to file cache asynchronously for key {key}: {e}"
            )

    async def _initialize_async_components(self):
        """
        Initializes asynchronous components of the StandardDecorator, including performance logging and resource profiling.
        This method ensures that all asynchronous initializations are completed before the decorator is used.
        """
        await self.initialize_performance_logger()
        await self._initialize_resource_profiling()

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

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logging.info(
                f"Async call to {func.__name__} with args: {args} and kwargs: {kwargs}"
            )
            if self.enable_validation:
                await self._validate_func_signature(func, *args, **kwargs)
            try:
                start_time = asyncio.get_event_loop().time()
                result = await self.wrapper_logic(func, True, *args, **kwargs)
                end_time = asyncio.get_event_loop().time()
                await self.log_performance(func, start_time, end_time)
                return result
            except Exception as e:
                logging.error(f"Exception in async call to {func.__name__}: {e}")
                raise
            finally:
                self.end_time = asyncio.get_event_loop().time()

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

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
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
        start_time = asyncio.get_event_loop().time()
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
                        result = await asyncio.to_thread(func, *args, **kwargs)

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
                        await asyncio.sleep(delay)
                    attempt += 1
                except Exception as e:
                    logging.error(f"Exception during {func.__name__} execution: {e}")
                    raise e
                finally:
                    # Performance monitoring
                    end_time = asyncio.get_event_loop().time()
                    execution_time = end_time - start_time
                    logging.info(
                        f"Execution of {func.__name__} completed in {execution_time:.2f}s"
                    )
                    await self.log_performance(func, start_time, end_time)

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

        try:
            # Retrieve the function's signature and bind the provided arguments
            sig = signature(func)
            logging.debug(f"Function signature: {sig}")
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            logging.debug(f"Bound arguments with defaults applied: {bound_args}")

            # Retrieve type hints and validate each argument against its expected type
            type_hints = get_type_hints(func)
            logging.debug(f"Type hints: {type_hints}")

            # Ensure thread safety with asyncio.Lock for argument validation
            async with self.validation_lock:
                for name, value in bound_args.arguments.items():
                    expected_type = type_hints.get(
                        name, None
                    )  # Set a default value of None
                    logging.debug(
                        f"Validating argument '{name}' with value '{value}' against expected type '{expected_type}'"
                    )
                    if expected_type is not None:
                        if asyncio.iscoroutinefunction(expected_type):
                            if not asyncio.iscoroutine(
                                value
                            ) and not asyncio.iscoroutinefunction(value):
                                raise TypeError(
                                    f"Argument '{name}' must be a coroutine or a coroutine function, got {type(value)}"
                                )
                        elif not isinstance(value, expected_type):
                            raise TypeError(
                                f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
                            )
        except Exception as e:
            logging.error(
                f"Error validating function signature for {func.__name__}: {e}"
            )
            raise
        finally:
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
        start_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} at {start_time}",
            extra={"async_mode": True},
        )

        try:
            # Determine the position of the argument in the function's signature
            parameters = list(signature(func).parameters)
            if arg_name not in parameters:
                raise ValueError(
                    f"Argument '{arg_name}' not found in {func.__name__}'s signature"
                )
            result = parameters.index(arg_name)

            # Log the determined position
            logging.debug(
                f"Argument position for {arg_name} in {func.__name__}: {result}",
                extra={"async_mode": True},
            )
        except Exception as e:
            logging.error(
                f"Error getting arg position for {arg_name} in {func.__name__}: {e}",
                exc_info=True,
                extra={"async_mode": True},
            )
            raise
        finally:
            # Log the completion of the process
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            logging.debug(
                f"Getting arg position for {arg_name} in {func.__name__} completed in {execution_time:.2f}s",
                extra={"async_mode": True},
            )
            # Ensure logging is performed asynchronously without blocking the event loop
            await asyncio.sleep(0)
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

        # Ensure thread safety with asyncio.Lock for argument validation
        async with self.validation_lock:
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

        # Ensure thread safety with asyncio.Lock for type validation
        async with self.validation_lock:
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
                            [
                                await self.validate_type(elem, element_type)
                                for elem in value
                            ]
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
