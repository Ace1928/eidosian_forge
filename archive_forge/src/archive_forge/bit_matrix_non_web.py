import asyncio
import logging
from typing import List, Tuple, Optional, Union, Any
from functools import wraps
import cProfile
import pstats
import io
import tracemalloc
import signal
import sys
from contextlib import contextmanager
from memory_profiler import profile
import cachetools.func
from typing import TypeVar, Callable
import os
import json
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T")

# Initialize logging with both file and console handlers
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bit_matrix.log"),
        logging.StreamHandler(),
    ],
)

# Initialize profiling
pr = cProfile.Profile()
pr.enable()

# Initialize memory tracing
tracemalloc.start()


# Initialize signal handler for graceful termination
def signal_handler(sig, frame):
    logging.info("Received signal to terminate. Dumping logs and profiles.")
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    logging.info(s.getvalue())
    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Current memory usage: {current / 10**6}MB; Peak: {peak / 10**6}MB")
    tracemalloc.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Context manager for profiling code blocks
@contextmanager
def profiled():
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    logging.info(s.getvalue())


# Decorator for wrapping synchronous functions to be run asynchronously
def async_wrap(func: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs) -> T:
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = profile(func)
        with profiled():
            return await loop.run_in_executor(executor, pfunc, *args, **kwargs)

    return run


# Decorator for caching results of expensive functions
def cache(maxsize=128, typed=False):
    def decorator(func):
        cache_path = f"{func.__name__}_cache.json"
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
        else:
            cache_data = {}

        @cachetools.func.lru_cache(maxsize=maxsize, typed=typed)
        def cached_func(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache_data:
                result = func(*args, **kwargs)
                cache_data[key] = result
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
            return cache_data[key]

        @wraps(func)
        def wrapper(*args, **kwargs):
            return cached_func(*args, **kwargs)

        return wrapper

    return decorator


@async_wrap
@cache(maxsize=1024)
def input_optional(prompt: str) -> Optional[str]:
    """
    Prompt for optional input, return None if empty.

    Args:
        prompt (str): The prompt to display to the user.

    Returns:
            Optional[str]: The user's input, or None if the input is empty.
    """
    logging.debug(f"Prompting user with: {prompt}")
    response = input(prompt + " (leave blank if not applicable): ").strip()
    logging.debug(f"User response: {response}")
    return response if response else None


@async_wrap
@cache(maxsize=256)
def convert_to_binary(value: int, length: int = 4) -> str:
    """
    Convert a decimal value to binary with fixed length.

    Args:
        value (int): The decimal value to convert.
        length (int, optional): The desired length of the binary string. Defaults to 4.

    Returns:
        str: The binary representation of the decimal value, padded to the specified length.
    """
    logging.debug(f"Converting {value} to binary with length {length}")
    try:
        binary = format(value, f"0{length}b")
        logging.debug(f"Binary result: {binary}")
        return binary
    except ValueError as e:
        logging.exception(f"Error converting {value} to binary: {e}")
        raise


@async_wrap
@cache(maxsize=512)
async def validate_and_parse_input(
    prompt: str,
    expected_type: Callable[[str], T] = int,
    delimiter: str = ",",
    optional: bool = False,
) -> List[T]:
    """
    Prompt for input, validate, parse, and return a list of values of the expected type.

    Args:
        prompt (str): The prompt to display to the user.
        expected_type (Callable[[str], T], optional): The expected type of the input values. Defaults to int.
        delimiter (str, optional): The delimiter to split the input string. Defaults to ",".
        optional (bool, optional): Whether the input is optional. Defaults to False.

    Returns:
        List[T]: A list of parsed values of the expected type.

    Raises:
        ValueError: If the input cannot be parsed as the expected type.
    """
    while True:
        logging.debug(f"Validating and parsing input for prompt: {prompt}")
        response = await input_optional(prompt)
        if optional and response is None:
            logging.debug("Optional input not provided")
            return []
        try:
            values = [expected_type(item.strip()) for item in response.split(delimiter)]
            logging.debug(f"Parsed values: {values}")
            return values
        except ValueError as e:
            logging.exception(
                f"Invalid input. Please ensure your input matches the expected format: {expected_type.__name__}"
            )


@async_wrap
@cache(maxsize=128)
async def construct_bit_matrix(
    version: int,
    channels: List[int],
    counts: List[int],
    signs: List[int],
    figures: int,
    values: List[int],
    uncertainties: List[int],
) -> List[List[Union[int, str, List[Union[int, str]]]]]:
    """
    Construct the bit matrix representation.

    Args:
        version (int): The version of the bit matrix (1 for Left Handed, 2 for Right Handed).
        channels (List[int]): The list of channel values (1 for present, 0 for absent).
        counts (List[int]): The list of count values for each channel.
        signs (List[int]): The list of sign values for each count (0 for negative, 1 for positive).
        figures (int): The minimum length of significant figures for the values in count.
        values (List[int]): The list of values for the significant figures.
        uncertainties (List[int]): The list of uncertainty values for each channel in each count.

    Returns:
        List[List[Union[int, str, List[Union[int, str]]]]]: The constructed bit matrix.
    """
    logging.debug("Constructing bit matrix")
    version_matrix = [version]
    channel_matrix = channels
    count_matrix = [len(counts)] + counts
    sign_matrix = signs
    figures_matrix = [figures]
    value_matrix = [await convert_to_binary(val) for val in values]
    uncertainty_matrix = uncertainties
    bit_matrix = [
        version_matrix,
        channel_matrix,
        count_matrix,
        sign_matrix,
        figures_matrix,
        value_matrix,
        uncertainty_matrix,
    ]
    logging.debug(f"Constructed bit matrix: {bit_matrix}")
    return bit_matrix


@async_wrap
@cache(maxsize=64)
async def prompt_for_values() -> (
    Tuple[int, List[int], List[int], List[int], int, List[int], List[int]]
):
    """
    Prompt user for input and return a tuple of all components, with logic-based prompting.

    Returns:
        Tuple[int, List[int], List[int], List[int], int, List[int], List[int]]: A tuple containing:
            - version (int): The version of the bit matrix (1 for Left Handed, 2 for Right Handed).
            - channels (List[int]): The list of channel values (1 for present, 0 for absent).
            - counts (List[int]): The list of count values for each channel.
            - signs (List[int]): The list of sign values for each count (0 for negative, 1 for positive).
            - figures (int): The minimum length of significant figures for the values in count.
            - values (List[int]): The list of values for the significant figures.
            - uncertainties (List[int]): The list of uncertainty values for each channel in each count.
    """
    logging.debug("Prompting for values")
    version = await validate_and_parse_input("Enter version (1 for Left Handed, 2 for Right Handed): ", optional=True))[0] or 1
    channels = await validate_and_parse_input("Enter channels (comma-separated, 1 for present, 0 for absent): ", optional=True) or [1, 0, 0, 0, 0, 0, 0, 0, 0]
    counts = await validate_and_parse_input("Enter counts for each channel (comma-separated): ", optional=True) or [0]
    signs = await validate_and_parse_input("Enter signs for each count (comma-separated, 0 for negative, 1 for positive): ",optional=True,) or [1]
    figures = await validate_and_parse_input("Enter minimum length of significant figures (default 6): ", optional=True) or 6
    values = await validate_and_parse_input("Enter values for significant figures (comma-separated, 0-9): ", optional=True) or [0]
    uncertainties = await validate_and_parse_input("Enter uncertainties for each channel in each count (comma-separated, 0 for negative, 1 for positive for Version 1; reverse for Version 2): ",optional=True,) or [0]
    logging.debug(f"Prompted values: version={version}, channels={channels}, counts={counts}, signs={signs}, figures={figures}, values={values}, uncertainties={uncertainties}")
    
    return version, channels, counts, signs, figures, values, uncertainties


async def main():
    """Main function to drive the program."""
    logging.info("Starting main function")
    version, channels, counts, signs, figures, values, uncertainties = await prompt_for_values()
    bit_matrix = await construct_bit_matrix(version, channels, counts, signs, figures, values, uncertainties)
    print("Constructed Bit Matrix Representation:")
    for part in bit_matrix:
        print(part)
    logging.info("Main function completed")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    finally:
        loop.close()
        logging.info("Program completed.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    finally:
        loop.close()
        logging.info("Program completed.")
