import numpy as np
from typing import Tuple, List

"""
CRITICALLY IMPORTANT DETAILS:
# Floating Point Precision

# 32-bit floating point numbers have a precision of 23 bits.
Decimal Precision≈Binary_Precision×log10(2)
Substituting the binary precision of 23 bits for a 32 bit float:
Decimal_Precision≈23×log10(2)≈6.92
Thus, a precision of 6 bits is chosen for the fractional part of the number as the 7th bit is unreliable.

# 64-bit floating point numbers have a precision of 52 bits.
Decimal Precision≈Binary_Precision×log10(2)
Substituting the binary precision of 52 bits for a 64 bit float:
Decimal Precision≈53×log10(2)≈53×0.3010≈15.953
Thus, a precision of 15 bits is chosen for the fractional part of the number as the 16th bit is unreliable.

#Scientific Notation
# 32-bit floating point numbers have a range of approximately 10^-38 to 10^38.
Single Precision (32-bit)
Format: 1 bit for sign, 8 bits for exponent, and 23 bits for the fraction (mantissa).
Bias: 127
Exponent Range: -126 to +127 (after bias adjustment)
Precision: Approximately 7 decimal places.
In single precision, the number is represented as:
(−1)sign×1.fraction×2exponent+−127 
where the "1." before the fraction part is the implicit leading bit (normalized form). The precision of about 7 decimal places comes from the 23 bits allocated for the fraction part.

# 64-bit floating point numbers have a range of approximately 10^-308 to 10^308.
Double Precision (64-bit)
Format: 1 bit for sign, 11 bits for exponent, and 52 bits for the fraction (mantissa).
Bias: 1023
Exponent Range: -1022 to +1023 (after bias adjustment)
Precision: Approximately 16 decimal places.
In double precision, the number is represented as:
(−1)sign×1.fraction×2exponent+−1023
where the "1." before the fraction part is the implicit leading bit (normalized form). The precision of about 16 decimal places comes from the 52 bits allocated for the fraction part.

# Real World Degrees of Freedom
    - 1. X dimension # Length corresponds to the x-axis.
    - 2. Y dimension # Width corresponds to the y-axis.
    - 3. Z dimension # Height corresponds to the z-axis.
    - 4. T dimension # Time corresponds to the t-axis.
    - 5. X Rotation # Roll corresponds to the orientation in radians around the x-axis.
    - 6. Y Rotation # Pitch corresponds to the orientation in radians around the y-axis.
    - 7. Z Rotation # Yaw corresponds to the orientation in radians around the z-axis.
    - 8. T Rotation # Time corresponds to the orientation in radians around the t-axis.
    - 9. X Scale Factor # X Scale corresponds to the scaling factor along the x-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 10. Y Scale Factor # Y Scale corresponds to the scaling factor along the y-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 11. Z Scale Factor # Z Scale corresponds to the scaling factor along the z-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 12. T Scale Factor # T Scale corresponds to the scaling factor along the t-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 13. X Curvature # X Skew corresponds to the curvature along the x-axis. From 0 to -1 for concave, 0 - 1 for convex.
    - 14. Y Curvature # Y Skew corresponds to the curvature along the y-axis. From 0 to -1 for concave, 0 - 1 for convex.
    - 15. Z Curvature # Z Skew corresponds to the curvature along the z-axis. From 0 to -1 for concave, 0 - 1 for convex.
    - 16. T Curvature # T Skew corresponds to the curvature along the t-axis. From 0 to -1 for concave, 0 - 1 for convex.


# Bit Matrix Representation of Floating Point Numbers
# Arbitrary Precision - Precision Affects Matrix Size
# 2 versions of system.
# Each version producing a different matrix representation and requiring different operations.
# Kind of akin to chirality in chemistry and biology and physics.
    1. 0 for negative, 1 for positive; 0 for absent, 1 for present. (Left Handed)
    2. 1 for negative, 0 for positive; 1 for absent, 0 for present. (Right Handed)
Overall Representation: {
    ([Version(1 value)], # Maximum Number of Values = 1
    [Channel(9 values)], # Maximum Number of Values = 9
    [Count((16xChannel(for each value not none)) values)], # Maximum Number of Values = 144 
    [Sign((9xCount(for each not none) values)],  # Maximum Number of Values = 9 x 144 = 1296
    [Figures(min length of values in count, default 6)],  # Maximum Number of Values = 1
    [Value(For each sig figuresxnumber Channelsxnumber of Count and each Value 0 - 9 recorded as binary mapping)],  # Maximum Number of Values = 6 x 9 x 144 x 10 = 77760
    [Uncertainty( for each of the channels in each of the counts based on the last figure present. For Version 1, 0 for negative, 1 for positive. For Version 2, 1 for negative, 0 for positive.)]  # Maximum Number of Values = 1296 (one for each of the possible values that make up all possible channels and all possible counts)
}

# Version: 1 for left handed, 2 for right handed.
# Channel: For all parts of the value from 1 real up to 8 (octonion) imaginary parts. 1 for present, 0 for absent.
# Count: The number of components that make up the matrix (number of vectors) maximum of 16, for each channel, corresponding to comprehensive real world degrees of freedom. 
# Sign: For version 1(L): 0 for negative, 1 for positive. For version 2(R): 1 for negative, 0 for positive.
# Figures: The count for the minimum length of sig figures for the values in count OR a default of 6.
# Value: The value of the sig figs in count.
# Uncertainty: The uncertainty of the value of the sig figs in count.
# Total Bits:
    Single Vector Single Dimension negative or positive:
    Value to Convert: -1.23456
        Version: 1
        Channels: 1 (only 1 real)
        Count: 1 (only 1 vector)
        Sign: 0 (only 1 sign and 1 channel and for version 1 0 corresponds to negative)
        Figures: 6 (6 significant figures in value) binary mapping for 6: 0110
        Value (binary mappy matrix for each sig fig): [(0,0,0,0), (1,0,0,1), (0,0,1,1), (0,1,0,0), (1,0,1,0), (1,1,0,0)]   
        Uncertainty: Last digit > 5 and Version = 1 so Uncertainty = 1
    Final Matrix: [
        ([(1)],[(1,0,0,0,0,0,0,0,0)],[(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)],[(0)],[(0110)],[(0,0,0,0),(1,0,0,1),(0,0,1,1),(0,1,0,0),(1,0,1,0),(1,1,0,0)],[(1)])
        ]

        
This is a further refinement of the logic
The representation might not be perfect
Claude Ultra Modern Version
"""
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
from contextlib import asynccontextmanager, contextmanager
from memory_profiler import profile
import cachetools.func
from cachetools import TTLCache
import aiofiles
import aiohttp
from aiohttp import web
import json
from datetime import datetime

# Configure logging with maximum verbosity
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s - %(levelname)s - %(module)s - %(process)d - %(thread)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "bit_matrix.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "verbose",
        },
        "errors": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "bit_matrix_errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "verbose",
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
logging.config.dictConfig(log_config)

# Initialize profiling
pr = cProfile.Profile()
pr.enable()

# Initialize memory tracing
tracemalloc.start()

# Initialize cache
cache = TTLCache(maxsize=1024, ttl=3600)

# Initialize web app
app = web.Application()


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


# Async context manager for profiling code blocks
@asynccontextmanager
async def profiled():
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        logging.info(s.getvalue())


# Decorator for wrapping synchronous functions to be run asynchronously
def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = profile(func)
        async with profiled():
            return await loop.run_in_executor(executor, pfunc, *args, **kwargs)

    return run


# Decorator for caching results of expensive functions
def cache_result(key: str, maxsize: int = 128, ttl: int = 600, typed: bool = False):
    def decorator(func):
        @cachetools.func.ttl_cache(maxsize=maxsize, ttl=ttl, typed=typed)
        async def cached_func(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await cached_func(*args, **kwargs)
            except Exception as e:
                logging.exception(
                    f"Error in {func.__name__} with args {args} and kwargs {kwargs}"
                )
                raise e

            async with aiofiles.open("bit_matrix_cache.json", mode="r") as f:
                data = json.loads(await f.read())
            data[key] = result
            async with aiofiles.open("bit_matrix_cache.json", mode="w") as f:
                await f.write(json.dumps(data, default=str))
            return result

        return wrapper

    return decorator


@async_wrap
@cache_result(key="input_optional", maxsize=1024)
async def input_optional(prompt: str) -> Optional[str]:
    """Prompt for optional input, return None if empty."""
    logging.debug(f"Prompting user with: {prompt}")
    response = input(prompt + " (leave blank if not applicable): ").strip()
    logging.debug(f"User response: {response}")
    return response if response else None


@async_wrap
@cache_result(key="convert_to_binary", maxsize=256)
async def convert_to_binary(value: int, length: int = 4) -> str:
    """Convert a decimal value to binary with fixed length."""
    logging.debug(f"Converting {value} to binary with length {length}")
    try:
        binary = format(int(value), f"0{length}b")
    except (ValueError, TypeError) as e:
        logging.exception(f"Error converting {value} to binary: {e}")
        raise e
    logging.debug(f"Binary result: {binary}")
    return binary


@async_wrap
@cache_result(key="validate_and_parse_input", maxsize=512)
async def validate_and_parse_input(
    prompt: str, expected_type: Any = int, delimiter: str = ",", optional: bool = False
) -> List:
    """Prompt for input, validate, parse, and return a list of values of the expected type."""
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
        except (ValueError, TypeError) as e:
            logging.exception(
                f"Invalid input. Please ensure your input matches the expected format. Error: {e}"
            )


@async_wrap
@cache_result(key="construct_bit_matrix", maxsize=128)
async def construct_bit_matrix(
    version: int,
    channels: List[int],
    counts: List[int],
    signs: List[int],
    figures: int,
    values: List[int],
    uncertainties: List[int],
) -> List:
    """Construct the bit matrix representation."""
    logging.debug("Constructing bit matrix")
    try:
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
    except Exception as e:
        logging.exception(f"Error constructing bit matrix: {e}")
        raise e
    logging.debug(f"Constructed bit matrix: {bit_matrix}")
    return bit_matrix


@async_wrap
@cache_result(key="prompt_for_values", maxsize=64)
async def prompt_for_values() -> (
    Tuple[int, List[int], List[int], List[int], int, List[int], List[int]]
):
    """Prompt user for input and return a tuple of all components, with logic-based prompting."""
    logging.debug("Prompting for values")
    try:
        version = (
        await validate_and_parse_input(
            "Enter version (1 for Left Handed, 2 for Right Handed): ", optional=True
        )
    )[0] or 1
    channels = await validate_and_parse_input(
        "Enter channels (comma-separated, 1 for present, 0 for absent): ",
        optional=True,
    ) or [1, 0, 0, 0, 0, 0, 0, 0, 0]
    counts = await validate_and_parse_input(
        "Enter counts for each channel (comma-separated): ", optional=True
    ) or [0]
    signs = await validate_and_parse_input(
        "Enter signs for each count (comma-separated, 0 for negative, 1 for positive): ",
        optional=True,
    ) or [1]
    figures = (
        await validate_and_parse_input(
            "Enter minimum length of significant figures (default 6): ",
            optional=True,
        )
    )[0] or 6
    values = await validate_and_parse_input(
        "Enter values for significant figures (comma-separated, 0-9): ",
        optional=True,
    ) or [0]
    uncertainties = await validate_and_parse_input(
        "Enter uncertainties for each channel in each count (comma-separated, 0 for negative, 1 for positive for Version 1; reverse for Version 2): ",
        optional=True,
    ) or [0]

    logging.debug(
        f"Prompted values: version={version}, channels={channels}, counts={counts}, signs={signs}, figures={figures}, values={values}, uncertainties={uncertainties}"
    )
    return version, channels, counts, signs, figures, values, uncertainties
async def main():
    """Main function to drive the program."""
    



async def main():
    """Main function to drive the program."""
    try:
        logging.info("Starting main function")
        result = await prompt_for_values()
        version, channels, counts, signs, figures, values, uncertainties = result
        bit_matrix = await construct_bit_matrix(
            version, channels, counts, signs, figures, values, uncertainties
        )
        print("Constructed Bit Matrix Representation:")
        for part in bit_matrix:
            print(part)
    except Exception as e:
        logging.exception(f"Error in main function: {e}")
        raise e


# Web handler for bit matrix construction
async def construct_matrix_handler(request):
    try:
        data = await request.json()
        version = data.get("version", 1)
        channels = data.get("channels", [1, 0, 0, 0, 0, 0, 0, 0, 0])
        counts = data.get("counts", [0])
        signs = data.get("signs", [1])
        figures = data.get("figures", 6)
        values = data.get("values", [0])
        uncertainties = data.get("uncertainties", [0])

        bit_matrix = await construct_bit_matrix(
            version, channels, counts, signs, figures, values, uncertainties
        )
        return web.json_response({"bit_matrix": bit_matrix})
    except Exception as e:
        logging.exception(f"Error in construct matrix handler: {e}")
        return web.json_response({"error": str(e)}, status=400)


app.add_routes([web.post("/construct_matrix", construct_matrix_handler)])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    web.run_app(app)
"""     
        
The format of the bit matrix represenation allows for optimised efficient bitwise operations and comparisons.
The expected result being that the bit matrix representation can be used to perform operations on floating point numbers with arbitrary precision, allowing for more accurate calculations and comparisons and with greater efficiency than traditional floating point representations.
The key processes expected to make this achievable and significant are:
1. Conversion between floating point numbers and bit matrix representation.
2. Vector operations using bit matrix representation.
3. Comparison and logical operations using bit matrix representation.
4. Precision adjustment and error handling.
5. Integration with existing numerical and scientific computing libraries.
6. Potential for parallel processing and optimisation.
7. Potential for extension to multi-dimensional and multi-vector operations.
8. Potential for integration with machine learning and AI algorithms.
9. Potential for integration with quantum computing algorithms.
10. Potential for integration with blockchain and cryptographic algorithms.
11. Potential for integration with financial and economic algorithms.
12. Potential for integration with scientific and engineering algorithms.
13. Potential for integration with data analysis and visualisation algorithms.
14. Potential for integration with game development and simulation algorithms.
15. Potential for integration with educational and research algorithms.
16. Potential for integration with security and privacy algorithms.
17. Potential for integration with networking and communication algorithms.
18. Potential for integration with automation and robotics algorithms.
19. Potential for integration with healthcare and medical algorithms.
20. Potential for integration with environmental and sustainability algorithms.
21. Potential for integration with social and cultural algorithms.
22. Potential for integration with legal and governance algorithms.
23. Potential for integration with creative and artistic algorithms.
24. Potential for integration with historical and archival algorithms.
25. Potential for integration with future and speculative algorithms.
26. Potential for integration with interdisciplinary and cross-disciplinary algorithms.
"""


"""
Claude 3 Optimised version in chat window

"""
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

"""



"""


"""
Indego version from chat

"""


"""



"""

#####################################################################################
#                                                                                   #
#                             #Basic Vector Operations#                             #
#                                                                                   #
#####################################################################################


def vector_add(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Add two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    np.ndarray: The resultant vector after addition.
    """
    return np.add(v1, v2)


def vector_subtract(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Subtract two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    np.ndarray: The resultant vector after subtraction.
    """
    return np.subtract(v1, v2)


def vector_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the dot product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    float: The dot product of the vectors.
    """
    return np.dot(v1, v2)


def vector_multiply(v: np.ndarray, scalar: float) -> np.ndarray:
    """
    Multiply a vector by a scalar using numpy.

    Args:
    v (np.ndarray): Vector to multiply.
    scalar (float): Scalar to multiply by.

    Returns:
    np.ndarray: The resultant vector after multiplication.
    """
    return np.multiply(v, scalar)


def vector_divide(v: np.ndarray, scalar: float) -> np.ndarray:
    """
    Divide a vector by a scalar using numpy.

    Args:
    v (np.ndarray): Vector to divide.
    scalar (float): Scalar to divide by.

    Returns:
    np.ndarray: The resultant vector after division.
    """
    return np.divide(v, scalar)


#####################################################################################
#                                                                                   #
#                             #Complex Vector Operations#                           #
#                                                                                   #
#####################################################################################


def vector_rotate_2d(v: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a 2D vector by a given angle.

    Args:
        v (np.ndarray): 2D vector to rotate.
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated vector.
    """
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.dot(rotation_matrix, v)


def vector_rotate_3d(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a 3D vector around a specified axis by a given angle.

    Args:
        v (np.ndarray): 3D vector to rotate.
        axis (np.ndarray): Axis to rotate around.
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated vector.
    """
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product_matrix = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation_matrix = (
        cos_angle * np.eye(3)
        + sin_angle * cross_product_matrix
        + (1 - cos_angle) * np.outer(axis, axis)
    )
    return np.dot(rotation_matrix, v)


def vector_magnitude(v: np.ndarray) -> float:
    """
    Calculate the magnitude of a vector using numpy.

    Args:
    v (np.ndarray): Vector to calculate the magnitude of.

    Returns:
    float: The magnitude of the vector.
    """
    return np.linalg.norm(v)


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    float: The angle between the vectors in radians.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def vector_projection(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the projection of v1 onto v2 using numpy.

    Args:
    v1 (np.ndarray): Vector to project.
    v2 (np.ndarray): Vector to project onto.

    Returns:
    np.ndarray: The projection of v1 onto v2.
    """
    return np.dot(v1, v2) / np.dot(v2, v2) * v2


def vector_rejection(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the rejection of v1 from v2 using numpy.

    Args:
    v1 (np.ndarray): Vector to reject.
    v2 (np.ndarray): Vector to reject from.

    Returns:
    np.ndarray: The rejection of v1 from v2.
    """
    return v1 - vector_projection(v1, v2)


def reflect_vector(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Reflect a vector across a specified axis.

    Args:
        v (np.ndarray): Vector to reflect.
        axis (np.ndarray): Axis to reflect across, must be normalized.

    Returns:
        np.ndarray: Reflected vector.
    """
    return v - 2 * np.dot(v, axis) * axis


def vector_cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the cross product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    np.ndarray: The cross product of the vectors.
    """
    return np.cross(v1, v2)


def vector_triple_product(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Calculate the triple product of three vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    v3 (np.ndarray): Third vector.

    Returns:
    float: The triple product of the vectors.
    """
    return np.dot(v1, np.cross(v2, v3))


#####################################################################################
#                                                                                   #
#                         #Comparative Vector Operations#                           #
#                                                                                   #
#####################################################################################


def vector_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    float: The angle between the vectors in radians.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def vector_is_parallel(v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if two vectors are parallel using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    bool: True if the vectors are parallel, False otherwise.
    """
    return np.allclose(np.cross(v1, v2), np.zeros(3))


def vector_is_orthogonal(v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if two vectors are orthogonal using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    bool: True if the vectors are orthogonal, False otherwise.
    """
    return np.allclose(np.dot(v1, v2), 0)


def vector_is_normalized(v: np.ndarray) -> bool:
    """
    Check if a vector is normalized using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is normalized, False otherwise.
    """
    return np.allclose(np.linalg.norm(v), 1)


def vector_is_zero(v: np.ndarray) -> bool:
    """
    Check if a vector is zero using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is zero, False otherwise.
    """
    return np.allclose(v, np.zeros(v.shape))


def vector_is_equal(v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if two vectors are equal using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    bool: True if the vectors are equal, False otherwise.
    """
    return np.allclose(v1, v2)


def vector_is_orthonormal(v: np.ndarray) -> bool:
    """
    Check if a vector is orthonormal using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is orthonormal, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is a basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is a basis, False otherwise.
    """
    return np.linalg.matrix_rank(v) == v.shape[0]


def vector_is_linearly_independent(v: np.ndarray) -> bool:
    """
    Check if a vector is linearly independent using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is linearly independent, False otherwise.
    """
    return np.linalg.matrix_rank(v) == v.shape[0]


def vector_is_linearly_dependent(v: np.ndarray) -> bool:
    """
    Check if a vector is linearly dependent using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is linearly dependent, False otherwise.
    """
    return np.linalg.matrix_rank(v) < v.shape[0]


def vector_is_orthogonal_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthogonal basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthogonal basis, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_orthonormal_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthonormal basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthonormal basis, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_normalized_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is a normalized basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is a normalized basis, False otherwise.
    """
    return np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1]))


def vector_is_orthogonal_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthogonal matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthogonal matrix, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_orthonormal_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthonormal matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthonormal matrix, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_normalized_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is a normalized matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is a normalized matrix, False otherwise.
    """
    return np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1]))


def vector_is_orthogonal_basis_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthogonal basis matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthogonal basis matrix, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


#####################################################################################
#                                                                                   #
#                             #Multi-Vector Operations#                             #
#                                                                                   #
#####################################################################################


def vector_n_dot_product(v1: np.ndarray, v2: np.ndarray, n: int) -> float:
    """
    Calculate the nth dot product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    n (int): The nth dot product to calculate.

    Returns:
    float: The nth dot product of the vectors.
    """
    return np.dot(v1, v2) ** n


def vector_n_cross_product(v1: np.ndarray, v2: np.ndarray, n: int) -> np.ndarray:
    """
    Calculate the nth cross product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    n (int): The nth cross product to calculate.

    Returns:
    np.ndarray: The nth cross product of the vectors.
    """
    return np.cross(v1, v2) ** n


def vector_n_triple_product(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, n: int
) -> float:
    """
    Calculate the nth triple product of three vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    v3 (np.ndarray): Third vector.
    n (int): The nth triple product to calculate.

    Returns:
    float: The nth triple product of the vectors.
    """
    return np.dot(v1, np.cross(v2, v3)) ** n


def vector_dot_product_between_n_vectors(*vectors: List[float]) -> float:
    """
    Calculate the dot product between n vectors using numpy.

    Args:
    *vectors (List[float]): List of vectors.

    Returns:
    float: The dot product between the vectors.

    """
    return np.dot(*vectors)


def vector_normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to have a magnitude of 1.

    Args:
    v (np.ndarray): Vector to normalize.

    Returns:
    np.ndarray: Normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


#####################################################################################
#                                                                                   #
#                             #Vector Utilities#                                    #
#                                                                                   #
#####################################################################################


def adjust_precision_of_bit_matrix(bm: np.ndarray, new_precision: int) -> np.ndarray:
    """
    Adjust the precision of a bit matrix representation of vectors.

    Args:
        bm (np.ndarray): Bit matrix to adjust.
        new_precision (int): New precision level.

    Returns:
        np.ndarray: Adjusted bit matrix.
    """
    # Implementation would involve recalculating the bit matrix based on new precision.
    # This is a placeholder for the actual implementation.
    pass


def decompose_vector(v: np.ndarray, bases: List[np.ndarray]) -> List[np.ndarray]:
    """
    Decompose a vector into components along a set of orthogonal bases.

    Args:
        v (np.ndarray): Vector to decompose.
        bases (List[np.ndarray]): Orthogonal bases for decomposition.

    Returns:
        List[np.ndarray]: Components of the vector along the given bases.
    """
    components = []
    for base in bases:
        projection = np.dot(v, base) / np.dot(base, base) * base
        components.append(projection)
    return components


#####################################################################################
#                                                                                   #
#                #   Vector to Bit Matrix InterConversion   #                       #
#                                                                                   #
#####################################################################################


def float_to_bit_matrix(v: np.ndarray, precision: int = 6) -> np.ndarray:
    """
    Convert a floating point vector to a bit matrix based on the specified precision.
    Each number is represented by sign, value, and uncertainty bits.

    Args:
    v (np.ndarray): Vector of floats.
    precision (int): Number of bits dedicated to the fractional part.

    Returns:
    np.ndarray: The bit matrix representing the vector.
    """
    # Normalize the vector to the range -1 to 1 for sign bit implementation.
    normalized_v = v / np.max(np.abs(v))
    bit_matrix = np.array(
        [
            [1 if num < 0 else 0]  # Sign bit
            + [
                int(bit)
                for bit in f"{abs(num):0.{precision}f}".replace(".", "")[:precision]
            ]  # Value bits
            + [0]  # Uncertainty bit, currently simplified
            for num in normalized_v
        ]
    )
    return bit_matrix


def bit_matrix_to_float(bm: np.ndarray, original_max: float) -> np.ndarray:
    """
    Convert a bit matrix back to a vector of floats.

    Args:
    bm (np.ndarray): Bit matrix.
    original_max (float): Original maximum value used for normalization.

    Returns:
    np.ndarray: The resultant vector of floats.
    """

    def bits_to_float(bits: List[int]) -> float:
        # Reconstruct the floating point number from bits
        sign = -1 if bits[0] == 1 else 1
        value_str = "".join(str(bit) for bit in bits[1:-1])
        value = sign * float(f"0.{value_str}")
        return value * original_max

    return np.array([bits_to_float(bits) for bits in bm])


#####################################################################################
#                                                                                   #
#                        #   Bit Matrix Vector Operations   #                       #
#                                                                                   #
#####################################################################################


def bitwise_and_between_bit_matrices(bm1: np.ndarray, bm2: np.ndarray) -> np.ndarray:
    """
    Perform a bitwise AND operation between two bit matrices.

    Args:
        bm1 (np.ndarray): First bit matrix.
        bm2 (np.ndarray): Second bit matrix.

    Returns:
        np.ndarray: Resultant bit matrix after bitwise AND operation.
    """
    return np.bitwise_and(bm1, bm2)


def bitwise_or_between_bit_matrices(bm1: np.ndarray, bm2: np.ndarray) -> np.ndarray:
    """
    Perform a bitwise OR operation between two bit matrices.

    Args:
        bm1 (np.ndarray): First bit matrix.
        bm2 (np.ndarray): Second bit matrix.

    Returns:
        np.ndarray: Resultant bit matrix after bitwise OR operation.
    """
    return np.bitwise_or(bm1, bm2)


def invert_bit_matrix(bm: np.ndarray) -> np.ndarray:
    """
    Invert a bit matrix, switching all 0s to 1s and vice versa.

    Args:
        bm (np.ndarray): Bit matrix to invert.

    Returns:
        np.ndarray: Inverted bit matrix.
    """
    return np.logical_not(bm).astype(int)


def reduce_bit_matrix(bm: np.ndarray, factor: int) -> np.ndarray:
    """
    Reduce the size of a bit matrix by a specified factor, combining bits by averaging.

    Args:
        bm (np.ndarray): Bit matrix to reduce.
        factor (int): Reduction factor.

    Returns:
        np.ndarray: Reduced bit matrix.
    """
    # Placeholder for actual implementation
    pass




    # Placeholder for actual implementation
    pass

    # Placeholder for actual implementation
    pass

    Returns:
        np.ndarray: Reduced bit matrix.
    """
    # Placeholder for actual implementation
    pass


    pass

    # Placeholder for actual implementation
    pass

