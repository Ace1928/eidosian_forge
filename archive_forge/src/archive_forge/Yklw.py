import sys
import logging
from typing import List, Tuple, Generator
from functools import wraps
from time import time, perf_counter
from contextlib import contextmanager
from memory_profiler import profile
import multiprocessing
import argparse
import itertools
import functools
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Importing necessary modules for Unicode character handling, color formatting, performance profiling, resource management, parallel processing, and user interaction
from typing import List, Generator


def log_execution_time(func):
    """
    Decorator to log the execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {execution_time:.5f} seconds")
        return result

    return wrapper


def log_memory_usage_decorator(func):
    """
    Decorator to log the memory usage before and after executing a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        before = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        after = psutil.Process().memory_info().rss
        logging.info(
            f"Memory usage of {func.__name__}: {before} bytes -> {after} bytes"
        )
        return result

    return wrapper


# Define the function to programmatically generate characters using memory-efficient generators
@log_execution_time
@log_memory_usage_decorator
def generate_utf_characters(
    custom_ranges: List[Tuple[int, int]] = None
) -> Generator[str, None, None]:
    """
    Generates a comprehensive list of UTF block, pipe, shape, and other related characters by systematically iterating through Unicode code points using memory-efficient generators.

    Args:
        custom_ranges (List[Tuple[int, int]], optional): Custom Unicode ranges to include in the generation process. Defaults to None.

    Returns:
        Generator[str, None, None]: A generator yielding unique UTF characters including block, pipe, shape, and other related characters.
    """
    default_ranges: List[Tuple[int, int]] = [
        (0x2500, 0x257F),  # Box Drawing
        (0x2580, 0x259F),  # Block Elements
        (0x25A0, 0x25FF),  # Geometric Shapes
        (0x2600, 0x26FF),  # Miscellaneous Symbols
        (0x2700, 0x27BF),  # Dingbats
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    ]

    # Use custom ranges if provided, otherwise use default ranges
    unicode_ranges = custom_ranges if custom_ranges else default_ranges

    # Iterate through the defined Unicode ranges using memory-efficient generators
    for start, end in unicode_ranges:
        for code_point in range(start, end + 1):
            try:
                character = chr(code_point)  # Convert code point to character
                yield character  # Yield character from the generator
            except ValueError as e:
                logging.error(
                    f"Skipping invalid or non-representable code point: {code_point}. Error: {str(e)}"
                )


colors = [
    "\033[38;2;0;0;0m",  # Absolute Black
    # ... (Shades of Grey)
    "\033[38;2;255;255;255m",  # Pure White
    "\033[38;2;255;0;0m",  # Red
    # ... (Shades of Red)
    "\033[38;2;255;165;0m",  # Orange
    # ... (Shades of Orange)
    "\033[38;2;255;255;0m",  # Yellow
    # ... (Shades of Yellow)
    "\033[38;2;0;128;0m",  # Green
    # ... (Shades of Green)
    "\033[38;2;0;0;255m",  # Blue
    # ... (Shades of Blue)
    "\033[38;2;75;0;130m",  # Indigo
    # ... (Shades of Indigo)
    "\033[38;2;238;130;238m",  # Violet
    # ... (Shades of Violet)
]


# Define a context manager to log memory usage
@contextmanager
def log_memory_usage():
    """
    Context manager to log memory usage before and after a code block.
    """
    before = psutil.Process().memory_info().rss
    try:
        yield
    finally:
        after = psutil.Process().memory_info().rss
        logging.info(f"Memory usage: {before} bytes -> {after} bytes")


# Define a class to handle character printing with colors
class CharacterPrinter:
    """
    A class to handle printing characters with colors.
    """

    def __init__(self, characters: Generator[str, None, None], colors: List[str]):
        """
        Initialize the CharacterPrinter.

        Args:
            characters (Generator[str, None, None]): A generator yielding characters to print.
            colors (List[str]): A list of color codes to apply to the characters.
        """
        self.characters = characters
        self.colors = colors

    @log_execution_time
    @log_memory_usage_decorator
    def print_characters(self):
        """
        Print the characters with colors.
        """
        with log_memory_usage():
            for char, color in zip(self.characters, itertools.cycle(self.colors)):
                print(f"{color}{char}\033[0m", end=" ")
            print()  # Print a newline after all characters


# Define a function to print characters with colors using multiprocessing
@log_execution_time
@log_memory_usage_decorator
def print_characters_in_colors(batch: List[str], colors: List[str]):
    """
    Print a batch of characters with colors using multiprocessing.

    Args:
        batch (List[str]): A batch of characters to print.
        colors (List[str]): A list of color codes to apply to the characters.
    """
    for char, color in zip(batch, itertools.cycle(colors)):
        print(f"{color}{char}\033[0m", end=" ")
    print()  # Print a newline after the batch


# Define the main function to handle character generation and printing
@profile
def main():
    """
    The main function to handle character generation and printing.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate and print UTF block, pipe, shape, and other related characters."
    )
    parser.add_argument(
        "--ranges",
        nargs="+",
        type=str,
        default=[],
        help="Custom Unicode ranges in the format 'start-end' (e.g., '2500-257F')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Output destination (console or file)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="characters.txt",
        help="Output file name (default: characters.txt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for parallel processing (default: 1000)",
    )
    args = parser.parse_args()

    # Process custom Unicode ranges
    custom_ranges = []
    for range_str in args.ranges:
        start, end = map(lambda x: int(x, 16), range_str.split("-"))
        custom_ranges.append((start, end))

    # Generate UTF characters based on the provided ranges
    characters = generate_utf_characters(custom_ranges)

    # Print or save the characters based on the output option
    if args.output == "console":
        # Use parallel processing to speed up character printing
        with multiprocessing.Pool() as pool:
            batches = itertools.islice(characters, args.batch_size)
            pool.starmap(
                print_characters_in_colors, zip(batches, itertools.repeat(colors))
            )
    else:
        with open(args.file, "w", encoding="utf-8") as file:
            for char in characters:
                file.write(f"{char}\n")


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
