import universal_wrapper as wrapper_logic  # Containers the wrapper logic required for running the sync and async wrappers for dynamic function execution
import async_logging as alogging  # Contains an advanced AsyncLogger which uses resource profiling to also capture logged process performance that utilises a DualLogging Class and file handlers, resource handlers and cache for advanced robust logging
import function_input_validation as validate  # Contains the function input validation logic for ensuring the correct input types and values are passed to the functions
import async_cache as acache  # Contains the async cache logic for caching the results of functions to improve performance and reduce resource usage
import async_file_handler as afile  # Contains the async file handler logic for reading and writing files in an asynchronous manner
import asyncio
import functools
import signal
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

FuncType = Callable[..., Any]
ArgsType = Tuple[Any, ...]
KwargsType = Dict[str, Any]
CacheType = Dict[str, Any]
ExceptionType = Exception
SignalType = Union[int, signal.Signals]
FrameType = Optional[types.FrameType]

cache_lock: afile.Lock = afile.Lock()  # Ensures thread safety for cache operations
log_lock: afile.Lock = afile.Lock()  # Ensures thread safety for file operations
file_io_lock: afile.Lock = afile.Lock()

if asyncio.get_event_loop().is_running():
    asyncio.run_coroutine_threadsafe(
    logger(): logger = alogging.logger,  # Initialize the logger
    validate_result(): bool = validate._validate_func_signature(
        func, args, kwargs
    )  # Validate the function signature
    start_time(): float = alogging.start_time()  # Get the start time
    end_time(): float = alogging.end_time()  # Get the end time
    log_performance(): None = alogging.log_performance(
        func, start_time, end_time
    )  # Log the performance of the function
    result(): Any = wrapper_logic.wrapper_logic(
        func, True, args, kwargs
    )  # Execute the function
    )
else: # Event Loop Not Running
    asyncio.run(
    logger(): alogging.AsyncLogger = alogging.AsyncLogger("SyncAsyncLauncher")
    validate_result(): bool = validate._validate_func_signature(
        func, args, kwargs
    )  # Validate the function signature
    start_time(): float = alogging.start_time()  # Get the start time
    end_time(): float = alogging.end_time()  # Get the end time
    log_performance(): None = alogging.log_performance(
        func, start_time, end_time
    )  # Log the performance of the function
    result(): Any = wrapper_logic.wrapper_logic(
        func, True, args, kwargs
    )  # Execute the function
    )

    

class SyncAsyncLauncher:
    """
    Description:
        This Class is responsible for launching the synchronous and asynchronous versions of the trading bot. It uses the UniversalWrapper class to wrap the functions and the AsyncLogger class to log the performance of the functions. It also uses the ResourceProfiler class to log the resource usage of the functions. The SyncAsyncLauncher class is designed to handle both synchronous and asynchronous functions seamlessly in a non blocking manner.
        It will be extended further to handle more advanced concurrency access and usage.
        It is built to be robust and flexible and dynamic and comprehensively logged.
        The resource and performance logging is now down via the async logger
        The wrapper logic is contained in the universal wrapper
        The Validation functions are contained in the function input validation module
        This class handles the dynamic and flexible execution of functions utilizing the utilities listed and ensuring graceful shutdown and signal management.
        It is designed to be robust and flexible and dynamic and comprehensively logged with explicit and robust error handling also incorporated in the advanced async logging module.
        It is essentially a universal coordinator for a program to utilize to allow for efficient and flexible and robust application execution.
        It will be further incorporated into a universal decorator that can be added to any function in any program and take it from synchronous and potentially inefficient execution to an asynchronous, caching, logged, validated, error handled, concurrency managed, resource profiled, performance logged, signal managed, graceful shutdown enabled function.
        Everything refactored into distinct classes to ensure easy flexibility and maintainability itself.
        Once complete it will be able to be run on a copy of itself to improve and optimize its own performance and resource usage.

    """

    def __init__(self) -> None:
        """
        Initializes the SyncAsyncLauncher class. It sets up the logger and the cache and the file cache path and the file I/O lock.
        """
        self.enable_caching: bool = True
        self.cache: CacheType = acache.cache
        self.file_cache_path: str = acache.cache_path

    @functools.wraps(func)  # Preserve the original function signature
    async def async_wrapper(*args: ArgsType, **kwargs: KwargsType) -> Any:
        """
        This function is the async wrapper that is returned by the decorator. It executes the function asynchronously and handles exceptions and logging.
        Args:
            *args: The positional arguments of the function.
            **kwargs: The keyword arguments of the function.
        Returns:
            The result of the function execution.
        Purpose: To wrap the function and execute it asynchronously, handling exceptions and logging and performance monitoring and validation.
        """
        alogging.info(
            f"Async call to {function_name} with args: {args} and kwargs: {kwargs}"
        )
        validate_result  # Validate the function signature
        try:  # Try to execute the function
            start_time  # Get the start time
            result  # Execute the function
            end_time  # Get the end time
            log_performance  # Log the performance of the function
            return result  # Return the result
        except exception:  # Handle exceptions
            logger.error(f"Exception in async call to {function_name}: {exception}")
            raise  # Raise the exception
        finally:  # Finally
            end_time  # Get the end time
            log_performance  # Log the performance of the function

    @functools.wraps(func)  # Preserve the original function signature
    def sync_wrapper(
        *args: ArgsType, **kwargs: KwargsType
    ) -> Any:  # Define the synchronous wrapper
        alogging.info(
            f"Sync call to {function_name} with args: {args} and kwargs: {kwargs}"
        )
        if asyncio.get_event_loop().is_running():  # Check if the event loop is running
            asyncio.run_coroutine_threadsafe(validate_result, asyncio.get_event_loop())
        else:
            asyncio.run(validate_result)

        try:  # Try to execute the function
            if (
                asyncio.get_event_loop().is_running()
            ):  # Check if the event loop is running
                start_time = asyncio.run_coroutine_threadsafe(
                    alogging.start_time(), asyncio.get_event_loop()
                ).result()  # Get the start time
                result = asyncio.run_coroutine_threadsafe(
                    func(*args, **kwargs), asyncio.get_event_loop()
                ).result()  # Execute the function
                end_time = asyncio.run_coroutine_threadsafe(
                    alogging.end_time(), asyncio.get_event_loop()
                ).result()  # Get the end time
            else:
                start_time = asyncio.run(alogging.start_time())  # Get the start time
                result = asyncio.run(func(*args, **kwargs))  # Execute the function
                end_time = asyncio.run(alogging.end_time())  # Get the end time
            if (
                asyncio.get_event_loop().is_running()
            ):  # Check if the event loop is running
                asyncio.run_coroutine_threadsafe(
                    alogging.log_performance(func, start_time, end_time),
                    asyncio.get_event_loop(),
                )  # Log the performance of the function
            else:
                asyncio.run(
                    alogging.log_performance(func, start_time, end_time)
                )  # Log the performance of the function
            return result  # Return the result
        except Exception as e:  # Handle exceptions
            alogging.error(f"Exception in sync call to {func.__name__}: {e}")
            raise  # Raise the exception

    async def async_sync_wrapper(func: FuncType) -> Union[FuncType, Callable[..., Any]]:
        """
        This function is a decorator that can be used to wrap a function and execute it either synchronously or asynchronously based on the function type. It logs the function call, validates the input arguments, and logs the performance of the function. It also handles exceptions and logs them appropriately. This function is designed to work with both synchronous and asynchronous functions seamlessly.
        Inserting logging, error handling and performance logging into the decorated function and
        executing dynamic/flexible validation on the input arguments to ensure the function is executed correctly.
        Args:
            func: The function to be wrapped and executed.
        Returns:
            The wrapped function that can be executed either synchronously or asynchronously.
        """
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    def _signal_handler(self, signum: SignalType, frame: FrameType) -> None:
        """
        Signal handler for initiating graceful shutdown. This method is designed to be compatible with asynchronous
        operations and ensures that the shutdown process is handled properly in an asyncio context.
        Args:
            signum: The signal number received.
            frame: The current stack frame.
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
        if self.cache is not None:
            async with self.file_io_lock:
                async with afile.open(self.file_cache_path, "wb") as f:
                    await f.write(acache.pickle.dumps(self.cache))
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
        tasks: List[asyncio.Task] = [
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
