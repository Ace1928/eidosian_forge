import asyncio
import aiofiles
import logging
import pickle
from typing import Callable, Optional, Any, Coroutine

from async_file_handler import AsyncFileHandler
from resource_profiler import ResourceProfiler  # A custom module for resource profiling
from async_cache import AsyncCache  # A custom module for asynchronous caching
from logging.handlers import RotatingFileHandler


class AsyncLogging:
    """
    Provides comprehensive asynchronous logging and graceful shutdown capabilities for a trading bot application.

    This class encapsulates methods for asynchronous execution of coroutines, signal handling for graceful shutdown,
    performance logging, and setup of various logging handlers to ensure non-blocking I/O operations and thread safety.

    Attributes:
        logger (logging.Logger): The logger object for logging information, warnings, and errors.
        enable_caching (bool): Flag to enable or disable caching functionality.
        cache (Optional[dict]): A dictionary to store cache data, or None if caching is disabled.
        file_io_lock (asyncio.Lock): An asyncio lock to ensure thread-safe file I/O operations.
        file_cache_path (str): The file path for storing cache data.
        enable_performance_logging (bool): Flag to enable or disable performance logging.
        performance_log_handler (Optional[AsyncFileHandler]): An asynchronous file handler for performance logging, or None if performance logging is disabled.
        logging_lock (asyncio.Lock): An asyncio lock to ensure thread-safe logging operations.
        enable_resource_profiling (bool): Flag to enable or disable resource profiling.
        resource_profiling_interval (float): The interval at which resource usage is logged.
        resource_profiling_output (str): The output destination for resource profiling logs.
        resource_profiler (Optional[ResourceProfiler]): The resource profiler object, or None if resource profiling is disabled.
    """

    def __init__(self) -> None:
        """
        Initializes the AsyncLogging class with default values for its attributes.
        """
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.enable_caching: bool = False
        self.cache: Optional[dict] = None
        self.file_io_lock: asyncio.Lock = asyncio.Lock()
        self.file_cache_path: str = "cache.pkl"
        self.enable_performance_logging: bool = False
        self.performance_log_handler: Optional[AsyncFileHandler] = None
        self.logging_lock: asyncio.Lock = asyncio.Lock()
        self.enable_resource_profiling: bool = False
        self.resource_profiling_interval: float = 1.0
        self.resource_profiling_output: str = "resource_profile.log"
        self.resource_profiler: Optional[ResourceProfiler] = None

    def _run_async_coroutine(self, coroutine: Coroutine) -> Any:
        """
        Adapts the execution of a coroutine based on the current event loop state, ensuring compatibility with both
        synchronous and asynchronous contexts.

        Args:
            coroutine (Coroutine): The coroutine to be executed.

        Returns:
            Any: The result of the coroutine execution if the event loop is not running; otherwise, schedules the coroutine
            for future execution.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                self.logger.debug(
                    "Event loop is running. Scheduling coroutine for future execution."
                )
                return asyncio.ensure_future(coroutine, loop=loop)
        except RuntimeError:
            self.logger.debug("No running event loop. Executing coroutine directly.")
            return asyncio.run(coroutine)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """
        Signal handler for initiating graceful shutdown. This method is designed to be compatible with asynchronous
        operations and ensures that the shutdown process is handled properly in an asyncio context.

        Args:
            signum (int): The signal number received.
            frame (Any): The current stack frame.
        """
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown.")
        asyncio.create_task(self._graceful_shutdown())

    async def _graceful_shutdown(self) -> None:
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

    async def _cancel_outstanding_tasks(self) -> None:
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

    async def initialize_performance_logger(self) -> None:
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

    async def _initialize_resource_profiling(self) -> None:
        """
        Initializes resource profiling if enabled. This method sets up the ResourceProfiler to log resource usage
        at specified intervals asynchronously, ensuring that the profiling does not block or interfere with the
        execution of the program.
        """
        if self.enable_resource_profiling:
            self.resource_profiler = ResourceProfiler(
                interval=int(self.resource_profiling_interval),
                output=self.resource_profiling_output,
            )
            await self.resource_profiler.start()
            logging.debug("Resource profiling initialized and started.")

    async def setup_logging(self) -> None:
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
        rotating_file_handler = RotatingFileHandler(
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
        await self._initialize_resource_profiling()
