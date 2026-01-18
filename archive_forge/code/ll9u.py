import sys
import logging
from typing import List, Tuple, Optional
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
Module: Block and Pipe Unicode Visualizer
Path: scripts/fun_stuff/block_and_pipe_unicode.py
Description: This module is dedicated to generating and displaying a comprehensive list of UTF block, pipe, and shape characters across a spectrum of colors.
Overview: The script systematically iterates through Unicode code points to generate characters, which are then displayed in a meticulously defined color spectrum.
Purpose: To enhance the visual representation and understanding of UTF block, pipe, and shape characters in various color contexts.
Scope: The script is intended for developers, designers, and enthusiasts interested in Unicode characters and their visual representation.
Definitions: UTF characters refer to characters defined in the Unicode Standard, which encompasses characters from the world's scripts, symbols, and emojis.
Key Features: 
- Comprehensive Unicode character generation
- Detailed color spectrum representation
- Systematic and exhaustive approach to character and color listing
Usage: Run the script in a Python 3.12 environment to see the characters displayed in the terminal.
Dependencies: Python 3.12
References: Unicode Standard Documentation
Authorship: [Your Name]
Versioning Details: Version 1.0.0
Functionalities: Character generation, color spectrum definition, character display in colors
Notes: Ensure terminal supports ANSI escape codes for color display.
Change Log: N/A
License: MIT License
Tags: Unicode, Characters, Visualization, Python
Contributors: [Your Name]
Security Considerations: N/A
Privacy Considerations: N/A
Performance Benchmarks: N/A
Limitations: Terminal must support ANSI escape codes for proper display.
TODO: 
- Add support for more Unicode character ranges
- Enhance color spectrum for more detailed representation
"""


# Define the function to programmatically generate characters
async def generate_utf_characters() -> List[str]:
    """
    Asynchronously generates a list of UTF block, pipe, and shape characters by systematically iterating through Unicode code points.

    Returns:
        List[str]: A list of unique UTF characters including block, pipe, and shape characters.
    """
    characters: List[str] = []  # Initialize an empty list to store the characters

    # Define the ranges of Unicode code points for block, pipe, and shape characters
    unicode_ranges: List[Tuple[int, int]] = [
        (0x2500, 0x257F),  # Box Drawing
        (0x2580, 0x259F),  # Block Elements
        (0x25A0, 0x25FF),  # Geometric Shapes
    ]

    # Iterate through the defined Unicode ranges
    for start, end in unicode_ranges:
        for code_point in range(start, end + 1):
            try:
                character = chr(code_point)  # Convert code point to character
                characters.append(character)  # Add character to the list
            except ValueError as e:
                # Log any ValueError exceptions encountered during character conversion
                logging.error(
                    f"Skipping invalid or non-representable code point: {code_point} - Error: {e}"
                )

    return characters  # Return the list of generated characters


# Define a comprehensive list of colors to represent the full spectrum
async def define_color_spectrum() -> List[Tuple[int, int, int]]:
    """
    Asynchronously defines a comprehensive list of colors to represent the full spectrum from black to white, including all shades and colors in between.

    Returns:
        List[Tuple[int, int, int]]: A list of RGB color tuples representing the full color spectrum.
    """
    colors: List[Tuple[int, int, int]] = (
        [(0, 0, 0)]  # Absolute Black
        + [(i, i, i) for i in range(1, 256)]  # Incrementally increasing shades of grey
        + [(255, i, 0) for i in range(0, 256)]  # Red to Orange spectrum
        + [(255, 255, i) for i in range(0, 256)]  # Orange to Yellow spectrum
        + [(255 - i, 255, 0) for i in range(0, 256)]  # Yellow to Green spectrum
        + [(0, 255, i) for i in range(0, 256)]  # Green to Blue spectrum
        + [(0, 255 - i, 255) for i in range(0, 256)]  # Blue to Indigo spectrum
        + [(i, 0, 255) for i in range(0, 256)]  # Indigo to Violet spectrum
        + [(255, i, 255) for i in range(0, 256)]  # Violet to White spectrum
        + [(255, 255, 255)]  # Pure White
    )
    return colors


async def print_characters_in_colors(
    characters: List[str], colors: List[Tuple[int, int, int]]
) -> None:
    """
    Asynchronously prints each character in the provided list in a series of colors.

    Parameters:
        characters: A list of characters to be printed.
        colors: A list of RGB color tuples.
    """
    for char in characters:
        logging.info(f"Character: {char} - Unicode: {ord(char)}")
        for color in colors:
            # ANSI escape code for color formatting
            print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{char}\033[0m", end=" ")
        print()  # Newline after printing all colors for a character


async def main():
    """
    Main function to orchestrate the asynchronous generation and printing of UTF characters in colors.
    """
    characters = await generate_utf_characters()
    colors = await define_color_spectrum()
    await print_characters_in_colors(characters, colors)


if __name__ == "__main__":
    asyncio.run(main())
