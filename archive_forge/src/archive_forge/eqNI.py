"""
convert_model.py

Author: [Author Name]
Creation Date: [Creation Date]

This module provides functionalities to convert RWKV models to a format that allows for faster loading and reduced CPU RAM usage. It includes command-line argument parsing and model conversion based on the specified strategy.

Functions:
- get_args: Parses command-line arguments.
- convert_model: Converts the RWKV model based on provided arguments.

"""

import argparse
import logging
from typing import NoReturn
from rwkv.model import RWKV

# Configure logging to adhere to best practices and improve readability
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_args() -> argparse.Namespace:
    """
    Parses and returns command line arguments for model conversion.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="convert_model",
        description="Convert RWKV model for faster loading and reduced CPU RAM usage.",
    )
    # Use 'input' and 'output' as argument flags to avoid confusion and adhere to conventions
    parser.add_argument(
        "--input",
        dest="input_model",
        type=str,
        metavar="INPUT",
        help="Filename for the input model.",
        required=True,
    )
    parser.add_argument(
        "--output",
        dest="output_model",
        type=str,
        metavar="OUTPUT",
        help="Filename for the output model.",
        required=True,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy for conversion. Enclose in quotes if it contains spaces or special characters. Refer to https://pypi.org/project/rwkv/ for format.",
        required=True,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppresses normal output, only displaying errors.",
    )
    return parser.parse_args()


def convert_model() -> None:
    """
    Converts an RWKV model based on the provided command line arguments.
    Enhanced error handling and logging for better maintainability.
    """
    try:
        args = get_args()
        # Improved logging for better clarity and troubleshooting
        if not args.quiet:
            # Directly pass the Namespace object to improve code readability and maintainability
            logging.info(f"Starting model conversion with parameters: {args}")
        RWKV.convert_and_save(
            input_model=args.input_model,
            strategy=args.strategy,
            verbose=not args.quiet,
            output_model=args.output_model,
        )
    except Exception as e:
        logging.error("An error occurred during model conversion: %s", e, exc_info=True)
        with open("error.txt", "w") as f:
            f.write(f"Error: {e}\n")


if __name__ == "__main__":
    convert_model()
# TODO:
# - Implement progress tracking for model conversion.
# - Add support for additional model formats.
# - Enhance error handling with more specific exceptions.
# - Consider integrating a logging configuration file for more flexible logging setups.
