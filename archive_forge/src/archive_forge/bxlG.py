import asyncio
import functools
import logging
import time
from inspect import iscoroutinefunction, signature, get_type_hints
from typing import Any, Callable, Dict, Tuple, TypeVar, Optional

# Type variable for the decorator
F = TypeVar("F", bound=Callable[..., Any])


class StandardDecorator:
    """
    A class encapsulating a decorator that enhances functions with logging, error handling,
    performance monitoring, automatic retrying on transient failures, optional result caching,
    and dynamic input validation and sanitization. Designed for use across the Neuro Forge INDEGO (EVIE) project.

    Attributes:
        retries (int): Number of times to retry the function on failure.
        delay (int): Delay between retries in seconds.
        cache_results (bool): Whether to cache the function's return value.
        log_performance (bool): Whether to log the performance of the function.
        validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Custom validation rules for function arguments.
        cache (Dict[Tuple, Any]): A cache to store function results if caching is enabled.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: int = 2,
        cache_results: bool = False,
        log_performance: bool = True,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
    ):
        """
        Initializes the StandardDecorator with the provided configuration.

        Args:
            retries (int): Number of times to retry the function on failure.
            delay (int): Delay between retries in seconds.
            cache_results (bool): Whether to cache the function's return value.
            log_performance (bool): Whether to log the performance of the function.
            validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Custom validation rules for function arguments.
        """
        self.retries = retries
        self.delay = delay
        self.cache_results = cache_results
        self.log_performance = log_performance
        self.validation_rules = validation_rules or {}
        self.cache = {}

    def __call__(self, func: F) -> F:
        """
        Makes the class instance callable, allowing it to be used as a decorator.

        Args:
            func (F): The function to be decorated.

        Returns:
            F: The decorated function, which may be either synchronous or asynchronous.
        """

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
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
                    result = await func(*args, **kwargs)
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
