import asyncio
import functools
import logging
import time
from inspect import iscoroutinefunction, signature
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    TypeVar,
    Optional,
    Type,
    get_type_hints,
)
from functools import lru_cache

# Type variable for the decorator
F = TypeVar("F", bound=Callable[..., Any])


class StandardDecorator:
    """
    A class encapsulating a decorator that enhances functions with logging, error handling,
    performance monitoring, automatic retrying on transient failures, optional result caching,
    and dynamic input validation and sanitization. Designed for use across the Neuro Forge INDEGO (EVIE) project.

    This decorator provides a robust framework for enhancing function execution with features like
    automatic retries on specified exceptions, performance logging, input validation, and result caching
    with a thread-safe LRU cache strategy. It allows for granular control over logging levels and
    includes support for complex validation rules, making it highly customizable and adaptable to various needs.

    Example usage:
        @StandardDecorator(retries=2, delay=1, log_level=logging.INFO, cache_maxsize=100)
        def my_function(param1):
            # Function body

    Attributes:
        retries (int): Number of times to retry the function on failure.
        delay (int): Delay between retries in seconds.
        cache_results (bool): Whether to cache the function's return value.
        log_level (int): Logging level for performance and error logs.
        validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Custom validation rules for function arguments.
        retry_exceptions (Tuple[Type[BaseException], ...]): Exceptions that trigger a retry.
        cache_maxsize (int): Max size for caching, using LRU strategy if enabled.
        enable_performance_logging (bool): Whether to log the performance of the function.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: int = 2,
        cache_results: bool = False,
        log_level: int = logging.DEBUG,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        cache_maxsize: int = 128,
        enable_performance_logging: bool = True,
    ):
        """
        Initializes the StandardDecorator with the provided configuration.

        Args:
            retries (int): Number of times to retry the function on failure.
            delay (int): Delay between retries in seconds.
            cache_results (bool): Whether to cache the function's return value.
            log_level (int): Logging level for performance and error logs.
            validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Custom validation rules for function arguments.
            retry_exceptions (Tuple[Type[BaseException], ...]): Exceptions that trigger a retry.
            cache_maxsize (int): Max size for caching, using LRU strategy if enabled.
            enable_performance_logging (bool): Whether to log the performance of the function.
        """
        self.retries = retries
        self.delay = delay
        self.cache_results = cache_results
        self.log_level = log_level
        self.validation_rules = validation_rules or {}
        self.retry_exceptions = retry_exceptions
        self.cache_maxsize = cache_maxsize
        self.enable_performance_logging = enable_performance_logging
        self.cache = {}

    def __call__(self, func: F) -> F:
        """
        Makes the class instance callable, allowing it to be used as a decorator. This method checks if the
        function is asynchronous or synchronous and wraps it with the appropriate logic to implement retries,
        caching, logging, and input validation.

        Args:
            func (F): The function to be decorated.

        Returns:
            F: The decorated function, which may be either synchronous or asynchronous.
        """
        # Apply LRU cache to the function if caching is enabled
        if self.cache_results:
            func = lru_cache(maxsize=self.cache_maxsize)(func)

        # Define an asynchronous wrapper for coroutine functions
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Validate inputs before executing the function
            self.validate_inputs(func, *args, **kwargs)
            # Generate a unique key for caching based on function arguments
            key = (args, frozenset(kwargs.items()))
            # Check if the result is already cached and return it if so
            if self.cache_results and key in self.cache:
                logging.log(
                    self.log_level,
                    f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}",
                )
                return self.cache[key]

            # Initialize attempt counter for retries
            attempts = 0
            while attempts < self.retries:
                try:
                    # Record start time for performance logging
                    start_time = time.time()
                    # Execute the decorated function
                    result = await func(*args, **kwargs)
                    # Record end time for performance logging
                    end_time = time.time()
                    # Log the execution time if performance logging is enabled
                    if self.enable_performance_logging:
                        logging.log(
                            self.log_level,
                            f"{func.__name__} executed in {end_time - start_time:.2f}s",
                        )
                    # Cache the result if caching is enabled
                    if self.cache_results:
                        self.cache[key] = result
                    return result
                # Catch only the specified retry exceptions
                except (
                    Exception if (Exception in self.retry_exceptions) else BaseException
                ) as e:
                    # Log the error and retry after the specified delay
                    logging.error(
                        f"{func.__name__} attempt {attempts + 1} failed with {e}, retrying..."
                    )
                    attempts += 1
                    await asyncio.sleep(self.delay)
            logging.debug(f"Final attempt for {func.__name__}")
            return await self.wrapper_logic(func, True, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            self.validate_inputs(func, *args, **kwargs)
            key = (args, frozenset(kwargs.items()))
            if self.cache_results and key in self.cache:
                logging.debug(
                    f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                return self.cache[key]
            return self.wrapper_logic(func, False, *args, **kwargs)

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    async def wrapper_logic(self, func: F, is_async: bool, *args, **kwargs) -> Any:
        """
        Contains the shared logic for both asynchronous and synchronous function wrappers.

        Args:
            func (F): The function being decorated.
            is_async (bool): Indicates whether the function is asynchronous.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Any: The result of executing the function, potentially with retries and caching.
        """
        self.validate_inputs(func, *args, **kwargs)
        key = (args, frozenset(kwargs.items()))
        if self.cache_results and key in self.cache:
            logging.debug(
                f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
            )
            return self.cache[key]

        attempts = 0
        while attempts < self.retries:
            try:
                start_time = time.time()
                result = (
                    await func(*args, **kwargs) if is_async else func(*args, **kwargs)
                )
                end_time = time.time()
                if self.log_performance:
                    logging.debug(
                        f"{func.__name__} executed in {end_time - start_time:.2f}s"
                    )
                if self.cache_results:
                    self.cache[key] = result
                return result
            except Exception as e:
                logging.error(
                    f"{func.__name__} attempt {attempts + 1} failed with {e}, retrying..."
                )
                attempts += 1
                if is_async:
                    await asyncio.sleep(self.delay)
                else:
                    time.sleep(self.delay)
        logging.debug(f"Final attempt for {func.__name__}")
        return await func(*args, **kwargs) if is_async else func(*args, **kwargs)

    def validate_inputs(self, func: F, *args: Any, **kwargs: Any) -> None:
        """
        Validates the inputs to the decorated function based on type hints and custom validation rules.

        Args:
            func (F): The function being decorated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If an argument does not match its type hint.
            ValueError: If an argument fails a custom validation rule.
        """
        bound_arguments = signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        arg_types = get_type_hints(func)

        for arg, value in bound_arguments.arguments.items():
            if arg in arg_types and not isinstance(value, arg_types[arg]):
                raise TypeError(f"Argument {arg} must be of type {arg_types[arg]}")
            if arg in self.validation_rules and not self.validation_rules[arg](value):
                raise ValueError(
                    f"Validation failed for argument {arg} with value {value}"
                )


# Existing imports and StandardDecorator class definition...

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Synchronous example function with input validation
    @StandardDecorator(
        retries=2,
        delay=1,
        log_level=logging.INFO,
        validation_rules={"x": lambda x: x > 0},
    )
    def sync_example(x: int) -> int:
        if x == 5:
            raise ValueError("Example of a retry scenario.")
        return x * 2

    # Asynchronous example function with caching
    @StandardDecorator(
        cache_results=True, cache_maxsize=2, enable_performance_logging=True
    )
    async def async_example(x: int) -> int:
        await asyncio.sleep(1)  # Simulate an I/O operation
        return x**2

    # Complex synchronous function demonstrating validation and error handling
    @StandardDecorator(
        retries=1,
        delay=2,
        log_level=logging.DEBUG,
        validation_rules={"text": lambda t: isinstance(t, str)},
    )
    def complex_sync_example(text: str, repeat: int) -> str:
        if repeat < 1:
            raise ValueError("repeat must be greater than 0")
        return text * repeat

    # Complex asynchronous function to test retry and caching with obfuscation
    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    async def complex_async_example(x: int) -> int:
        if x % 2 == 0:
            raise Exception("Even numbers simulate transient failures.")
        await asyncio.sleep(2)  # Simulate a longer I/O operation
        return x + 10

    # Running synchronous examples
    try:
        print(sync_example(5))
    except Exception as e:
        print(f"Sync example with retry failed: {e}")

    print(sync_example(10))

    try:
        print(complex_sync_example("Test", 0))
    except Exception as e:
        print(f"Complex sync example failed: {e}")

    print(complex_sync_example("Repeat", 3))

    # Running asynchronous examples
    async def run_async_examples():
        try:
            print(await async_example(3))
            print(await async_example(3))  # This should hit the cache
        except Exception as e:
            print(f"Async example failed: {e}")

        try:
            print(await complex_async_example(4))
        except Exception as e:
            print(f"Complex async example with even number failed: {e}")

        print(await complex_async_example(5))

    asyncio.run(run_async_examples())
