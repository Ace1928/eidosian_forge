import sys

# This module is designed to print a list of unique UTF block and pipe characters
# in various colors covering a spectrum including the rainbow colors, white, black,
# and multiple shades of grey and each color. This enhances the visual representation
# and understanding of these characters in different color contexts.

# Importing necessary modules for color representation
from typing import List, Tuple

# Define a list of unique UTF block characters and pipe characters
# ensuring no redundancy and covering a wide range of shapes.
characters: List[str] = [
    "\u2500",
    "\u2501",
    "\u2502",
    "\u2503",
    "\u2504",
    "\u2505",
    "\u2506",
    "\u2507",
    "\u2508",
    "\u2509",
    "\u250A",
    "\u250B",
    "\u250C",
    "\u250D",
    "\u250E",
    "\u250F",
    "\u2510",
    "\u2511",
    "\u2512",
    "\u2513",
    "\u2514",
    "\u2515",
    "\u2516",
    "\u2517",
    "\u2518",
    "\u2519",
    "\u251A",
    "\u251B",
    "\u251C",
    "\u251D",
    "\u251E",
    "\u251F",
    "\u2520",
    "\u2521",
    "\u2522",
    "\u2523",
    "\u2524",
    "\u2525",
    "\u2526",
    "\u2527",
    "\u2528",
    "\u2529",
    "\u252A",
    "\u252B",
    "\u252C",
    "\u252D",
    "\u252E",
    "\u252F",
    "\u2530",
    "\u2531",
    "\u2532",
    "\u2533",
    "\u2534",
    "\u2535",
    "\u2536",
    "\u2537",
    "\u2538",
    "\u2539",
    "\u253A",
    "\u253B",
    "\u253C",
    "\u253D",
    "\u253E",
    "\u253F",
    "\u2540",
    "\u2541",
    "\u2542",
    "\u2543",
    "\u2544",
    "\u2545",
    "\u2546",
    "\u2547",
    "\u2548",
    "\u2549",
    "\u254A",
    "\u254B",
    "\u254C",
    "\u254D",
    "\u254E",
    "\u254F",
    "\u2550",
    "\u2551",
    "\u2552",
    "\u2553",
    "\u2554",
    "\u2555",
    "\u2556",
    "\u2557",
    "\u2558",
    "\u2559",
    "\u255A",
    "\u255B",
    "\u255C",
    "\u255D",
    "\u255E",
    "\u255F",
    "\u2560",
    "\u2561",
    "\u2562",
    "\u2563",
    "\u2564",
    "\u2565",
    "\u2566",
    "\u2567",
    "\u2568",
    "\u2569",
    "\u256A",
    "\u256B",
    "\u256C",
    "\u256D",
    "\u256E",
    "\u256F",
    "\u2570",
    "\u2571",
    "\u2572",
    "\u2573",
    "\u2574",
    "\u2575",
    "\u2576",
    "\u2577",
    "\u2578",
    "\u2579",
    "\u257A",
    "\u257B",
    "\u257C",
    "\u257D",
    "\u257E",
    "\u257F",
    "\u2580",
    "\u2581",
    "\u2582",
    "\u2583",
    "\u2584",
    "\u2585",
    "\u2586",
    "\u2587",
    "\u2588",
    "\u2589",
    "\u258A",
    "\u258B",
    "\u258C",
    "\u258D",
    "\u258E",
    "\u258F",
    "\u2590",
    "\u2591",
    "\u2592",
    "\u2593",
    "\u2594",
    "\u2595",
    "\u2596",
    "\u2597",
    "\u2598",
    "\u2599",
    "\u259A",
    "\u259B",
    "\u259C",
    "\u259D",
    "\u259E",
    "\u259F",
    "\u25A0",
    "\u25A1",
    "\u25A2",
    "\u25A3",
    "\u25A4",
    "\u25A5",
    "\u25A6",
    "\u25A7",
    "\u25A8",
    "\u25A9",
    "\u25AA",
    "\u25AB",
    "\u25AC",
    "\u25AD",
    "\u25AE",
    "\u25AF",
    "\u25B0",
    "\u25B1",
    "\u25B2",
    "\u25B3",
    "\u25B4",
    "\u25B5",
    "\u25B6",
    "\u25B7",
    "\u25B8",
    "\u25B9",
    "\u25BA",
    "\u25BB",
    "\u25BC",
    "\u25BD",
    "\u25BE",
    "\u25BF",
    "\u25C0",
    "\u25C1",
    "\u25C2",
    "\u25C3",
    "\u25C4",
    "\u25C5",
    "\u25C6",
    "\u25C7",
    "\u25C8",
    "\u25C9",
    "\u25CA",
    "\u25CB",
    "\u25CC",
    "\u25CD",
    "\u25CE",
    "\u25CF",
    "\u25D0",
    "\u25D1",
    "\u25D2",
    "\u25D3",
    "\u25D4",
    "\u25D5",
    "\u25D6",
    "\u25D7",
    "\u25D8",
    "\u25D9",
    "\u25DA",
    "\u25DB",
    "\u25DC",
    "\u25DD",
    "\u25DE",
    "\u25DF",
    "\u25E0",
    "\u25E1",
    "\u25E2",
    "\u25E3",
    "\u25E4",
    "\u25E5",
    "\u25E6",
    "\u25E7",
    "\u25E8",
    "\u25E9",
    "\u25EA",
    "\u25EB",
    "\u25EC",
    "\u25ED",
    "\u25EE",
    "\u25EF",
    "\u25F0",
    "\u25F1",
    "\u25F2",
    "\u25F3",
    "\u25F4",
    "\u25F5",
    "\u25F6",
    "\u25F7",
    "\u25F8",
    "\u25F9",
    "\u25FA",
    "\u25FB",
    "\u25FC",
    "\u25FD",
    "\u25FE",
    "\u25FF",
]

# Define a list of colors to represent the spectrum including shades of grey and colors
# This list progresses systematically from black, through shades of grey, and then through the color spectrum
# from red, orange, yellow, green, blue, indigo, violet, and finally to white, as granularly as possible.
colors: List[Tuple[int, int, int]] = (
    [
        (0, 0, 0),  # Black
        # Shades of Grey, systematically progressing from darker to lighter
    ]
    + [(i, i, i) for i in range(28, 225, 28)]
    + [
        # Red to Orange spectrum
    ]
    + [(255, i, 0) for i in range(0, 225, 28)]
    + [
        # Orange to Yellow spectrum
    ]
    + [(255, 255, i) for i in range(28, 225, 28)]
    + [
        # Yellow to Green spectrum
    ]
    + [(255 - i, 255, 0) for i in range(28, 225, 28)]
    + [
        # Green to Blue spectrum
    ]
    + [(0, 255, i) for i in range(28, 225, 28)]
    + [
        # Blue to Indigo spectrum
    ]
    + [(0, 255 - i, 255) for i in range(28, 113, 28)]
    + [
        # Indigo to Violet spectrum
    ]
    + [(i, 0, 255) for i in range(113, 225, 28)]
    + [
        # Violet to White spectrum
    ]
    + [(255, i, 255) for i in range(28, 225, 28)]
    + [
        (255, 255, 255),  # White
    ]
)


def print_characters_in_colors(
    characters: List[str], colors: List[Tuple[int, int, int]]
) -> None:
    """
    Prints each character in the provided list in a series of colors.

    :param characters: A list of characters to be printed.
    :param colors: A list of RGB color tuples.
    """
    for char in characters:
        print(f"Character: {char} - Unicode: {ord(char)}", end=" | ")
        for color in colors:
            # ANSI escape code for color formatting
            print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{char}\033[0m", end=" ")
        print()  # Newline after printing all colors for a character


# Execute the function to print characters in colors
print_characters_in_colors(characters, colors)
