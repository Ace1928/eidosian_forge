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
"""

import asyncio
import logging
import logging.config
from typing import List, Tuple, Optional, Union, Any
import functools
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
        # Filter out the 'optional' keyword argument if present
        kwargs.pop(
            "optional", None
        )  # This line is added to remove 'optional' if it exists
        pfunc = functools.partial(func, *args, **kwargs)
        async with profiled():
            return await loop.run_in_executor(executor, pfunc)

    return run


# Decorator for caching results of expensive functions
def cache_result(key: str, maxsize: int = 128, ttl: int = 600, typed: bool = False):
    """
    Decorator to cache results of expensive async functions.

    Args:
        key (str): Unique key for caching results.
        maxsize (int): Maximum size of the cache.
        ttl (int): Time to live for cached results.
        typed (bool): Cache different types separately if True.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @cachetools.func.ttl_cache(maxsize=maxsize, ttl=ttl, typed=typed)
        async def cached_func(*args, **kwargs):
            """
            Async function wrapper to cache results.

            Args and Returns are dynamic as per the decorated function.
            """
            return await func(*args, **kwargs)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            """
            Wrapper function to handle caching logic and error handling.

            Args and Returns are dynamic as per the decorated function.
            """
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


# The following functions have been refined to ensure correct use of async/await,
# enhanced error handling, and appropriate caching and performance optimization.
# Each function includes comprehensive documentation and comments for clarity.


@async_wrap
@cache_result(key="input_optional", maxsize=1024)
async def input_optional(prompt: str) -> Optional[str]:
    """
    Prompt for optional input, return None if empty.

    Args:
        prompt (str): The prompt message to display to the user.

    Returns:
        Optional[str]: The user's response or None if empty.
    """
    logging.debug(f"Prompting user with: {prompt}")
    response = input(prompt + " (leave blank if not applicable): ").strip()
    logging.debug(f"User response: {response}")
    return response if response else None


@async_wrap
@cache_result(key="convert_to_binary", maxsize=256)
async def convert_to_binary(value: int, length: int = 4) -> str:
    """
    Convert a decimal value to binary with fixed length.

    Args:
        value (int): The decimal value to convert.
        length (int): The fixed length of the binary representation.

    Returns:
        str: The binary representation of the value.
    """
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
    """
    Prompt for input, validate, parse, and return a list of values of the expected type.

    Args:
        prompt (str): The prompt message for input.
        expected_type (Any): The expected type of each input value.
        delimiter (str): The delimiter used to separate input values.
        optional (bool): Whether the input is optional.

    Returns:
        List: A list of values of the expected type.
    """
    while True:
        logging.debug(f"Validating and parsing input for prompt: {prompt}")
        response = await input_optional(prompt)
        if optional and response is None:
            logging.debug("Optional input not provided")
            return []
        elif response is None:
            logging.debug("No input provided")
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
    """
    Construct the bit matrix representation.

    Args:
        version (int): Version of the matrix.
        channels (List[int]): List of channels.
        counts (List[int]): List of counts.
        signs (List[int]): List of signs.
        figures (int): Number of significant figures.
        values (List[int]): List of values.
        uncertainties (List[int]): List of uncertainties.

    Returns:
        List: The constructed bit matrix.
    """
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


# Web handler for bit matrix construction
async def construct_matrix_handler(request):
    """
    Asynchronously handles the request to construct a bit matrix.

    This function parses the request JSON for parameters, validates and processes them,
    then constructs and returns the bit matrix via a JSON response.

    Args:
        request: The request object from the web server.

    Returns:
        A web response object containing the bit matrix or an error message.
    """
    try:
        # Parse and validate request data
        data = await request.json()
        version = data.get("version", 1)
        channels = data.get("channels", [1, 0, 0, 0, 0, 0, 0, 0, 0])
        counts = data.get("counts", [0])
        signs = data.get("signs", [1])
        figures = data.get("figures", 6)
        values = data.get("values", [0])
        uncertainties = data.get("uncertainties", [0])

        # Construct the bit matrix using the validated data
        bit_matrix = await construct_bit_matrix(
            version, channels, counts, signs, figures, values, uncertainties
        )
        return web.json_response({"bit_matrix": bit_matrix})
    except Exception as e:
        # Log and respond with an error if any occurs during processing
        logging.exception(f"Error in construct matrix handler: {e}")
        return web.json_response({"error": str(e)}, status=400)


async def main():
    """
    Main function to drive the program.

    Sets up the web server routes and handles the main logic for constructing the bit matrix.
    """
    # Setup web server routes
    app.add_routes([web.post("/construct_matrix", construct_matrix_handler)])
    try:
        logging.info("Starting main function")
        # Await the coroutine for prompting values and ensure correct unpacking
        if (result := await prompt_for_values()) is not None:
            version, channels, counts, signs, figures, values, uncertainties = result
            # Construct the bit matrix with the obtained values
            bit_matrix = await construct_bit_matrix(
                version, channels, counts, signs, figures, values, uncertainties
            )
            print(f"Constructed Bit Matrix Representation: {bit_matrix}")
        else:
            logging.error("No result from prompt_for_values, unable to proceed.")
    except Exception as e:
        # Handle any exceptions that occur during the main function execution
        logging.exception(f"Error in main function: {e}")
        raise e


if __name__ == "__main__":
    # Initialize and run the event loop for the asynchronous main function
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # Start the web application
    web.run_app(app)
