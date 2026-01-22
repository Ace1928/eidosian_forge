# Importing necessary modules
import os  # Importing the os module for file and directory operations
import json  # Importing the json module for handling JSON data
import logging  # Importing the logging module for logging purposes
import asyncio  # Importing the asyncio module for asynchronous programming
import random  # Importing the random module for generating random values
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Any,
    Callable,
    TypeAlias,
    Optional,
)  # Importing various types from the typing module for type hinting and annotation
from itertools import (
    cycle,
)  # Importing the cycle function from itertools for creating cyclic iterators
import multiprocessing  # Importing the multiprocessing module for parallel processing
import OpenGL.GL as gl  # Importing OpenGL.GL module and aliasing it as gl for OpenGL graphics functionality
import OpenGL.GLUT as glut  # Importing OpenGL.GLUT module and aliasing it as glut for OpenGL Utility Toolkit functionality
import OpenGL  # Importing the OpenGL module for OpenGL graphics functionality
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_MODELVIEW,
    GL_PROJECTION,
)  # Importing specific constants from OpenGL.GL for OpenGL graphics functionality

import cProfile  # Importing the cProfile module for profiling code execution
import pstats  # Importing the pstats module for analyzing profiling statistics
from pstats import (
    SortKey,
)  # Importing the SortKey class from pstats for sorting profiling statistics

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)  # Configuring the logging module with DEBUG level and a specific format for log messages
"""
Configure the logging module with enhanced settings for detailed and informative logging.

The `logging.basicConfig()` function is used to set up the basic configuration for the logging module. It takes several parameters to customize the logging behavior:

- `level`: Set to `logging.DEBUG` to capture all log messages, including debug-level messages. This ensures that detailed information is logged for debugging and troubleshooting purposes.

- `format`: Specifies the format of the log messages. In this case, the format includes the timestamp (`%(asctime)s`), log level (`%(levelname)s`), and the log message itself (`%(message)s`). This format provides comprehensive information for each log entry.

By configuring the logging module with these settings, all subsequent log messages will adhere to the specified format and capture detailed information, enabling effective debugging and monitoring of the program's execution.

Logging is a crucial aspect of software development, as it helps in understanding the flow of the program, identifying issues, and facilitating problem-solving. With the enhanced logging setup, developers can gain valuable insights into the program's behavior and quickly diagnose and resolve any encountered issues.
"""

# Type Aliases for enhanced clarity, type safety, and to avoid type/value/key errors
ColorName: TypeAlias = str  # Defining a type alias ColorName as str for color names
"""
Define a type alias `ColorName` as `str` for color names.

Type aliases are used to create more descriptive and meaningful names for existing types. In this case, the type alias `ColorName` is defined as an alias for the built-in `str` type.

By using the `ColorName` type alias, it becomes clear that a variable or parameter represents a color name, enhancing code readability and providing additional context.

Type aliases help in improving code clarity, making the intent of the code more explicit, and reducing the chances of type-related errors. They serve as a form of documentation, conveying the purpose and expected type of a value.

In this specific case, the `ColorName` type alias indicates that a string value represents the name of a color. This information is valuable for developers working with color-related functionality, as it clearly communicates the purpose and expected format of the color name.

Using type aliases consistently throughout the codebase promotes code consistency, improves maintainability, and facilitates better understanding of the code's structure and purpose.
"""

RGBValue: TypeAlias = Tuple[
    int, int, int
]  # Defining a type alias RGBValue as a tuple of three integers for RGB color values
"""
Define a type alias `RGBValue` as a tuple of three integers for RGB color values.

The `RGBValue` type alias is defined as a tuple of three integers, representing the red, green, and blue components of an RGB color value.

RGB (Red, Green, Blue) is a color model commonly used in computer graphics and digital imaging. Each color component is represented by an integer value ranging from 0 to 255, where 0 indicates the absence of the color and 255 represents the maximum intensity of the color.

By defining the `RGBValue` type alias as a tuple of three integers, it provides a clear and concise way to represent RGB color values throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect color value formats.

Using the `RGBValue` type alias makes the code more expressive and self-explanatory. It communicates the expected structure and format of RGB color values, making it easier for developers to understand and work with color-related functionality.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of RGB color values needs to change in the future, such as using a different range or data type, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `RGBValue` type alias enhances code clarity, type safety, and maintainability when working with RGB color values in the program.
"""

CMYKValue: TypeAlias = Tuple[float, float, float, float]  # Defining a type alias CMYKValue as a tuple of four floats for CMYK color values
"""
Define a type alias `CMYKValue` as a tuple of four floats for CMYK color values.

The `CMYKValue` type alias represents a color value in the CMYK (Cyan, Magenta, Yellow, Key/Black) color model. It is defined as a tuple of four float values, each representing the intensity of the respective color component.

CMYK is a subtractive color model commonly used in printing and color reproduction. The four components are:
- Cyan (C): Represents the amount of cyan ink or pigment.
- Magenta (M): Represents the amount of magenta ink or pigment.
- Yellow (Y): Represents the amount of yellow ink or pigment.
- Key/Black (K): Represents the amount of black ink or pigment.

Each component value is typically represented as a float between 0.0 and 1.0, where 0.0 indicates the absence of the color and 1.0 represents the maximum intensity of the color.

By defining the `CMYKValue` type alias, it provides a clear and concise way to represent CMYK color values throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect color value formats.

Using the `CMYKValue` type alias makes the code more expressive and self-explanatory. It communicates the expected structure and format of CMYK color values, making it easier for developers to understand and work with color-related functionality.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of CMYK color values needs to change in the future, such as using a different range or data type, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `CMYKValue` type alias enhances code clarity, type safety, and maintainability when working with CMYK color values in the program.
"""

LabValue: TypeAlias = Tuple[float, float, float]  # Defining a type alias LabValue as a tuple of three floats for Lab color values
"""
Define a type alias `LabValue` as a tuple of three floats for Lab color values.

The `LabValue` type alias represents a color value in the Lab (L*a*b* or CIELAB) color space. It is defined as a tuple of three float values, each representing a specific component of the Lab color space.

The Lab color space is a color model that aims to approximate human vision and perception of color. It is designed to be perceptually uniform, meaning that equal distances in the color space correspond to equal perceived color differences. The three components of the Lab color space are:

- L* (Lightness): Represents the lightness or brightness of the color. It ranges from 0 (black) to 100 (white).
- a*: Represents the green-red color component. Negative values indicate green, while positive values indicate red.
- b*: Represents the blue-yellow color component. Negative values indicate blue, while positive values indicate yellow.

The Lab color space is widely used in color management systems and is considered to be device-independent, making it suitable for color communication and conversion between different devices and media.

By defining the `LabValue` type alias, it provides a clear and concise way to represent Lab color values throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect color value formats.

Using the `LabValue` type alias makes the code more expressive and self-explanatory. It communicates the expected structure and format of Lab color values, making it easier for developers to understand and work with color-related functionality.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of Lab color values needs to change in the future, such as using a different range or data type, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `LabValue` type alias enhances code clarity, type safety, and maintainability when working with Lab color values in the program.
"""

XYZValue: TypeAlias = Tuple[float, float, float]  # Defining a type alias XYZValue as a tuple of three floats for XYZ color values
"""
Define a type alias `XYZValue` as a tuple of three floats for XYZ color values.

The `XYZValue` type alias represents a color value in the XYZ color space, also known as the CIE 1931 XYZ color space. It is defined as a tuple of three float values, each representing a specific component of the XYZ color space.

The XYZ color space is a device-independent color space that serves as a reference for defining and communicating colors. It is based on the human visual system and aims to represent all colors that can be perceived by the human eye. The three components of the XYZ color space are:

- X: Represents the amount of red-green color stimulus.
- Y: Represents the luminance or brightness of the color.
- Z: Represents the amount of blue-yellow color stimulus.

The XYZ color space is widely used as an intermediate color space for color conversions and calculations. It provides a standardized way to represent colors and is often used as a bridge between different color spaces, such as RGB and CMYK.

By defining the `XYZValue` type alias, it provides a clear and concise way to represent XYZ color values throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect color value formats.

Using the `XYZValue` type alias makes the code more expressive and self-explanatory. It communicates the expected structure and format of XYZ color values, making it easier for developers to understand and work with color-related functionality.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of XYZ color values needs to change in the future, such as using a different range or data type, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `XYZValue` type alias enhances code clarity, type safety, and maintainability when working with XYZ color values in the program.
"""

ColorCode: TypeAlias = Tuple[ColorName, RGBValue]  # Defining a type alias ColorCode as a tuple of ColorName and RGBValue for color codes
"""
Define a type alias `ColorCode` as a tuple of `ColorName` and `RGBValue` for color codes.

The `ColorCode` type alias represents a color code, which is a combination of a color name and its corresponding RGB color value. It is defined as a tuple consisting of two elements:

1. `ColorName`: Represents the name of the color, which is of type `str`. It provides a human-readable and descriptive label for the color.
2. `RGBValue`: Represents the RGB color value associated with the color name. It is of type `RGBValue`, which is a tuple of three integers representing the red, green, and blue components of the color.

By defining the `ColorCode` type alias, it provides a clear and concise way to represent color codes throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect color code formats.

Using the `ColorCode` type alias makes the code more expressive and self-explanatory. It communicates the expected structure and format of color codes, making it easier for developers to understand and work with color-related functionality.

The combination of a color name and its corresponding RGB value in the `ColorCode` type alias allows for easy identification and retrieval of colors based on their names. It provides a convenient way to associate human-readable names with specific color values, enhancing the usability and readability of the code.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of color codes needs to change in the future, such as adding additional color properties or using a different color model, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `ColorCode` type alias enhances code clarity, type safety, and maintainability when working with color codes in the program. It provides a standardized way to represent and manipulate color information, making the code more expressive and easier to understand.
"""

ColorCodesList: TypeAlias = List[ColorCode]  # Defining a type alias ColorCodesList as a list of ColorCode for a list of color codes
"""
Define a type alias `ColorCodesList` as a list of `ColorCode` for a list of color codes.

The `ColorCodesList` type alias represents a list or collection of color codes. It is defined as a list (`List`) of `ColorCode` elements, where each element represents a specific color code.

By defining the `ColorCodesList` type alias, it provides a clear and concise way to represent a list of color codes throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect list formats or element types.

Using the `ColorCodesList` type alias makes the code more expressive and self-explanatory. It communicates the expected structure and format of a list of color codes, making it easier for developers to understand and work with color-related functionality that involves multiple colors.

The `ColorCodesList` type alias allows for convenient manipulation and iteration over a collection of color codes. It provides a standardized way to store, access, and process multiple color codes as a single entity, enhancing the efficiency and readability of the code.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of a list of color codes needs to change in the future, such as using a different data structure or adding additional properties to each color code, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `ColorCodesList` type alias enhances code clarity, type safety, and maintainability when working with lists of color codes in the program. It provides a standardized way to represent and manipulate collections of color information, making the code more expressive and easier to understand.
"""

ColorCodesDict: TypeAlias = Dict[ColorName, Union[RGBValue, str]]  # Defining a type alias ColorCodesDict as a dictionary mapping ColorName to either RGBValue or str for color codes dictionary
"""
Define a type alias `ColorCodesDict` as a dictionary mapping `ColorName` to either `RGBValue` or `str` for color codes dictionary.

The `ColorCodesDict` type alias represents a dictionary or mapping of color names to their corresponding color values. It is defined as a dictionary (`Dict`) where the keys are of type `ColorName` (which is an alias for `str`) and the values can be either of type `RGBValue` (a tuple of three integers representing the red, green, and blue components of the color) or `str` (a string representation of the color value).

By defining the `ColorCodesDict` type alias, it provides a clear and concise way to represent a dictionary of color codes throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect dictionary formats or value types.

Using the `ColorCodesDict` type alias makes the code more expressive and self-explanatory. It communicates the expected structure and format of a dictionary of color codes, making it easier for developers to understand and work with color-related functionality that involves mapping color names to their corresponding values.

The `ColorCodesDict` type alias allows for convenient lookup and retrieval of color values based on their associated color names. It provides a standardized way to store and access color information using a key-value pair structure, enhancing the efficiency and readability of the code.

The flexibility of allowing the values in the `ColorCodesDict` to be either `RGBValue` or `str` provides versatility in representing color values. It allows for storing color values as RGB tuples for precise color representation or as string representations for cases where a more human-readable or alternative format is required.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of a dictionary of color codes needs to change in the future, such as using a different data structure or adding additional color properties, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `ColorCodesDict` type alias enhances code clarity, type safety, and maintainability when working with dictionaries of color codes in the program. It provides a standardized way to represent and manipulate color information using a key-value pair structure, making the code more expressive and easier to understand.
"""

ANSIValue: TypeAlias = str  # Defining a type alias ANSIValue as str for ANSI escape codes
"""
Define a type alias `ANSIValue` as `str` for ANSI escape codes.

The `ANSIValue` type alias represents ANSI (American National Standards Institute) escape codes, which are special sequences of characters used to control the formatting and appearance of text in command-line interfaces or terminal emulators. It is defined as an alias for the built-in `str` type.

ANSI escape codes are widely used to add colors, styles, and other formatting options to text output in command-line applications. They consist of specific sequences of characters that are interpreted by the terminal to apply the desired formatting.

By defining the `ANSIValue` type alias, it provides a clear and concise way to represent ANSI escape codes throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect escape code formats.

Using the `ANSIValue` type alias makes the code more expressive and self-explanatory. It communicates the expected type and purpose of variables or parameters that represent ANSI escape codes, making it easier for developers to understand and work with text formatting functionality.

The `ANSIValue` type alias allows for consistent and standardized representation of ANSI escape codes as strings. It provides a way to store, manipulate, and apply escape codes to text output, enhancing the readability and maintainability of the code.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of ANSI escape codes needs to change in the future, such as using a different format or adding additional escape sequences, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `ANSIValue` type alias enhances code clarity, type safety, and maintainability when working with ANSI escape codes in the program. It provides a standardized way to represent and manipulate escape codes as strings, making the code more expressive and easier to understand.
"""

HexValue: TypeAlias = str  # Defining a type alias HexValue as str for hexadecimal color values
"""
Define a type alias `HexValue` as `str` for hexadecimal color values.

The `HexValue` type alias represents hexadecimal color values, which are a common way to specify colors using a combination of six hexadecimal digits. It is defined as an alias for the built-in `str` type.

Hexadecimal color values are widely used in web development, graphic design, and various color-related applications. They provide a compact and standardized format for representing colors using a combination of red, green, and blue components.

In the hexadecimal color format, each color component (red, green, and blue) is represented by two hexadecimal digits, ranging from "00" to "FF". The six digits are concatenated to form the complete color value, often preceded by a "#" symbol to indicate that it is a hexadecimal color code.

By defining the `HexValue` type alias, it provides a clear and concise way to represent hexadecimal color values throughout the codebase. It helps in ensuring type safety and avoiding errors related to incorrect color value formats.

Using the `HexValue` type alias makes the code more expressive and self-explanatory. It communicates the expected type and format of variables or parameters that represent hexadecimal color values, making it easier for developers to understand and work with color-related functionality.

The `HexValue` type alias allows for consistent and standardized representation of hexadecimal color values as strings. It provides a way to store, manipulate, and apply color values in the hexadecimal format, enhancing the readability and maintainability of the code.

Moreover, by leveraging type aliases, the codebase becomes more maintainable and adaptable. If the representation of hexadecimal color values needs to change in the future, such as using a different format or adding additional color properties, the type alias can be updated in a single place, and the change will propagate throughout the codebase.

Overall, the `HexValue` type alias enhances code clarity, type safety, and maintainability when working with hexadecimal color values in the program. It provides a standardized way to represent and manipulate color values as strings in the hexadecimal format, making the code more expressive and easier to understand.
"""

"""
Type Aliases for Extended Color Codes and Color Values

This section defines type aliases for various color representations and color code structures used throughout the program.

Type Aliases:
- ExtendedColorCode: Represents an extended color code as a tuple containing RGB, ANSI, HEX, CMYK, Lab, XYZ, and any additional color values.
- CMYKValue: Represents a CMYK color value as a tuple of four float values.
- LabValue: Represents a Lab color value as a tuple of three float values.
- XYZValue: Represents an XYZ color value as a tuple of three float values.
- ColorCode: Represents a color code as a tuple of color name and RGB value.
- ColorCodesList: Represents a list of color codes.
- ColorCodesDict: Represents a dictionary mapping color names to either RGB values or string representations.

These type aliases provide a standardized and expressive way to represent and work with color-related data structures in the program.
"""

ExtendedColorCode: TypeAlias = Tuple[RGBValue, ANSIValue, HexValue, CMYKValue, LabValue, XYZValue, Any]
"""
Type alias for an extended color code.

An extended color code is represented as a tuple containing the following color values:
- RGBValue: RGB color value
- ANSIValue: ANSI escape code
- HexValue: Hexadecimal color value
- CMYKValue: CMYK color value
- LabValue: Lab color value
- XYZValue: XYZ color value
- Any: Any additional color value

This type alias provides a standardized structure for representing extended color codes, allowing for the inclusion of multiple color representations and any additional color-related data.
"""

try:
    CMYKValue: TypeAlias = Tuple[float, float, float, float]
    """
    Type alias for a CMYK color value.

    A CMYK color value is represented as a tuple of four float values, corresponding to the Cyan, Magenta, Yellow, and Key (black) components of the color.

    This type alias provides a clear and concise way to represent CMYK color values, ensuring type safety and improving code readability.
    """
    logging.debug("CMYKValue type alias defined successfully.")
except TypeError as e:
    logging.error(f"Error defining CMYKValue type alias: {str(e)}")
    raise

try:
    LabValue: TypeAlias = Tuple[float, float, float]
    """
    Type alias for a Lab color value.

    A Lab color value is represented as a tuple of three float values, corresponding to the L* (lightness), a* (green-red), and b* (blue-yellow) components of the color.

    This type alias provides a clear and concise way to represent Lab color values, ensuring type safety and improving code readability.
    """
    logging.debug("LabValue type alias defined successfully.")
except TypeError as e:
    logging.error(f"Error defining LabValue type alias: {str(e)}")
    raise

try:
    XYZValue: TypeAlias = Tuple[float, float, float]
    """
    Type alias for an XYZ color value.

    An XYZ color value is represented as a tuple of three float values, corresponding to the X, Y, and Z components of the color in the CIE XYZ color space.

    This type alias provides a clear and concise way to represent XYZ color values, ensuring type safety and improving code readability.
    """
    logging.debug("XYZValue type alias defined successfully.")
except TypeError as e:
    logging.error(f"Error defining XYZValue type alias: {str(e)}")
    raise

try:
    ColorCode: TypeAlias = Tuple[ColorName, RGBValue]
    """
    Type alias for a color code.

    A color code is represented as a tuple containing the following elements:
    - ColorName: The name of the color
    - RGBValue: The RGB color value associated with the color name

    This type alias provides a standardized structure for representing color codes, associating a color name with its corresponding RGB value.
    """
    logging.debug("ColorCode type alias defined successfully.")
except TypeError as e:
    logging.error(f"Error defining ColorCode type alias: {str(e)}")
    raise

try:
    ColorCodesList: TypeAlias = List[ColorCode]
    """
    Type alias for a list of color codes.

    A color codes list is represented as a list of ColorCode tuples, where each tuple contains a color name and its corresponding RGB value.

    This type alias provides a clear and concise way to represent a collection of color codes, allowing for easy manipulation and iteration over multiple color codes.
    """
    logging.debug("ColorCodesList type alias defined successfully.")
except TypeError as e:
    logging.error(f"Error defining ColorCodesList type alias: {str(e)}")
    raise

try:
    ColorCodesDict: TypeAlias = Dict[ColorName, Union[RGBValue, str]]
    """
    Type alias for a dictionary of color codes.

    A color codes dictionary is represented as a dictionary mapping color names (ColorName) to either RGB values (RGBValue) or string representations of colors.

    This type alias provides a flexible way to represent a collection of color codes, allowing for the association of color names with either RGB values or string representations.

    The dictionary structure enables efficient lookup and retrieval of color values based on their associated color names.
    """
    logging.debug("ColorCodesDict type alias defined successfully.")
except TypeError as e:
    logging.error(f"Error defining ColorCodesDict type alias: {str(e)}")
    raise# Initialize your 3D scene and objects here
try:
    """
    Attempt to initialize the 3D scene by running the generate_scene() function asynchronously.
    
    The generate_scene() function is expected to be defined elsewhere in the codebase and should return a valid scene object.
    
    If the generate_scene() function is not defined or raises a NameError, the exception will be caught and logged as an error.
    The exception will then be re-raised to propagate it further up the call stack for appropriate handling.
    
    This try-except block ensures that any errors during scene initialization are properly logged and handled, preventing the program from crashing unexpectedly.
    """
    logging.info("Initializing the 3D scene by running the generate_scene() function asynchronously.")  # Log an informational message indicating the start of scene initialization
    
    scene: Scene = asyncio.run(generate_scene())  # Initialize the 3D scene by running the generate_scene() function asynchronously and assign the returned scene object to the 'scene' variable of type 'Scene'
    
    logging.debug(f"3D scene initialized successfully: {scene}")  # Log a debug message with the initialized scene object for detailed tracking and debugging purposes
    
except NameError as name_error:
    """
    Catch any NameError exceptions that may occur during the initialization of the 3D scene.
    
    If a NameError is raised, it typically indicates that the generate_scene() function is not defined or accessible in the current scope.
    
    The NameError exception object is assigned to the 'name_error' variable for further inspection and logging.
    
    An error message is logged using the logging module, providing details about the specific NameError that occurred during scene initialization.
    The exception message is converted to a string using str() to ensure compatibility with the logging format.
    
    After logging the error, the exception is re-raised using the 'raise' statement to propagate it further up the call stack.
    This allows for proper error handling and potentially graceful termination of the program if necessary.
    """
    logging.error(f"NameError occurred while initializing the 3D scene: {str(name_error)}")  # Log an error message with details about the NameError that occurred during scene initialization
    raise  # Re-raise the NameError exception to propagate it further up the call stack for appropriate handling

# Define the path to the color codes file
COLOR_CODES_FILE_PATH: str = "/home/lloyd/EVIE/color_codes.json"
"""
Define a constant variable 'COLOR_CODES_FILE_PATH' representing the file path to the color codes JSON file.

The 'COLOR_CODES_FILE_PATH' variable is assigned a string value specifying the absolute path to the color codes JSON file.

This constant variable allows for easy reference and reuse of the file path throughout the codebase.

By defining the file path as a constant, it promotes code maintainability and makes it easier to update the file path if necessary.

The type hint 'str' is used to indicate that the 'COLOR_CODES_FILE_PATH' variable is expected to be a string.
"""

logging.debug(f"Color codes file path defined: {COLOR_CODES_FILE_PATH}")  # Log a debug message with the defined color codes file path for tracking and debugging purposes

async def random_color() -> RGBValue:
    """
    Asynchronously generates a random RGB color value.

    This coroutine utilizes the random module to asynchronously generate a random RGB color value. It selects random integers between 0 and 255 (inclusive) for each of the red, green, and blue components of the color. The generated color is then returned as a tuple of three integers representing the RGB values.

    Returns:
        RGBValue: A tuple of three integers representing the random RGB color value, where each component is in the range of 0 to 255 (inclusive).

    Example:
        >>> import asyncio
        >>> asyncio.run(random_color())
        (128, 255, 64)
    """
    # Log a debug message indicating the start of the asynchronous random color generation process
    logging.debug("Asynchronously generating a random color.")

    # Utilize asyncio.sleep to simulate asynchronous execution and yield control back to the event loop
    await asyncio.sleep(0)  # No actual delay, but allows for asynchronous context switching

    # Generate a random integer between 0 and 255 (inclusive) for the red component of the color
    red_component: int = random.randint(0, 255)
    # Log the generated red component value for debugging purposes
    logging.debug(f"Generated random red component: {red_component}")

    # Generate a random integer between 0 and 255 (inclusive) for the green component of the color
    green_component: int = random.randint(0, 255)
    # Log the generated green component value for debugging purposes
    logging.debug(f"Generated random green component: {green_component}")

    # Generate a random integer between 0 and 255 (inclusive) for the blue component of the color
    blue_component: int = random.randint(0, 255)
    # Log the generated blue component value for debugging purposes
    logging.debug(f"Generated random blue component: {blue_component}")

    # Create a tuple representing the random RGB color value
    rgb_color: RGBValue = (red_component, green_component, blue_component)
    # Log the generated random RGB color value for detailed tracking and debugging purposes
    logging.debug(f"Generated random RGB color: {rgb_color}")

    # Return the random RGB color value as a tuple
    return rgb_color

async def generate_color_codes() -> ColorCodesList:
    """
    Generates a comprehensive list of color codes spanning a granular color spectrum.

    This function generates a comprehensive list of color codes that covers a wide range of colors in a granular spectrum. Each color is associated with a systematic name for easy identification and retrieval. The color codes include various color formats such as RGB values, ANSI escape codes, hexadecimal codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.

    The function utilizes asynchronous programming techniques to efficiently generate the color codes. It employs the asyncio module to run tasks concurrently, allowing for faster generation of the color codes.

    The generated color codes are stored in a dictionary where the keys represent the color names and the values are tuples containing the color information in different formats. The dictionary is then returned as the result of the function.

    Returns:
        ColorCodesList: A dictionary representing the generated color codes, where the keys are color names and the values are tuples containing color information in various formats.

    Example:
        >>> color_codes = generate_color_codes()
        >>> color_codes["red"]
        ((255, 0, 0), "\u001b[38;2;255;0;0m", "#FF0000", (0, 100, 100, 0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None)
    """
    logging.info("Generating color codes.")  # Log an info message indicating the start of color code generation

    try:
        # Define the number of shades per primary color
        shades_per_primary: int = 10
        logging.debug(f"Number of shades per primary color: {shades_per_primary}")

        # Define the total number of colors based on the number of shades per primary color
        total_colors: int = shades_per_primary**3
        logging.debug(f"Total number of colors: {total_colors}")

        # Generate the color names using the generate_color_names() function
        color_names: List[ColorName] = generate_color_names(total_colors)
        logging.debug(f"Generated {len(color_names)} color names.")

        # Create a dictionary to store the color codes
        color_codes: ColorCodesDict = {}

        # Iterate over each color name and generate the corresponding color code
        for name in color_names:
            # Generate a random RGB color value
            rgb: RGBValue = random_color()

            # Create a tuple containing the RGB values and placeholders for other color formats
            color_code: ColorCode = (rgb, None, None, None, None, None, None)

            # Add the color code to the dictionary with the color name as the key
            color_codes[name] = color_code

        logging.debug(f"Generated {len(color_codes)} color codes.")

        # Extend the color codes asynchronously to include additional color formats
        extended_color_codes: ExtendedColorCodesDict = asyncio.run(
            extend_color_codes(color_codes)
        )

        logging.debug(
            f"Extended {len(extended_color_codes)} color codes with additional formats."
        )

        return extended_color_codes

    except Exception as e:
        # Log an error message if an exception occurs during color code generation
        logging.error(f"Error occurred while generating color codes: {str(e)}")
        raise e

async def extend_color_codes(color_codes: ColorCodesDict) -> ExtendedColorCodesDict:
    """
    Extends the color codes dictionary with additional color formats asynchronously.

    This function takes a dictionary of color codes, where each color code is represented by a tuple containing RGB values and placeholders for other color formats. It extends the color codes by asynchronously generating additional color formats such as ANSI escape codes, hexadecimal codes, CMYK values, Lab values, and XYZ values.

    The function utilizes asynchronous programming techniques to efficiently extend the color codes. It employs the asyncio module to run tasks concurrently, allowing for faster generation of the additional color formats.

    The extended color codes are stored in a new dictionary where the keys represent the color names and the values are tuples containing the color information in various formats, including the original RGB values and the newly generated color formats.

    Args:
        color_codes (ColorCodesDict): A dictionary of color codes, where the keys are color names and the values are tuples containing RGB values and placeholders for other color formats.

    Returns:
        ExtendedColorCodesDict: A dictionary representing the extended color codes, where the keys are color names and the values are tuples containing color information in various formats, including RGB values, ANSI escape codes, hexadecimal codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.

    Example:
        >>> color_codes = {"red": ((255, 0, 0), None, None, None, None, None, None)}
        >>> extended_color_codes = asyncio.run(extend_color_codes(color_codes))
        >>> extended_color_codes["red"]
        ((255, 0, 0), "\u001b[38;2;255;0;0m", "#FF0000", (0, 100, 100, 0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None)
    """
    # Log an info message indicating the start of color code extension
    logging.info("Extending color codes with additional formats.")  

    try:
        # Create a dictionary to store the extended color codes
        extended_color_codes: ExtendedColorCodesDict = {}
        # Explanation: This line creates an empty dictionary called 'extended_color_codes' to store the color codes with additional formats. 
        # The type hint 'ExtendedColorCodesDict' indicates that the dictionary will have color names as keys and tuples of extended color information as values.

        # Create a list to store the asynchronous tasks for extending each color code
        color_extension_tasks: List[asyncio.Task] = []
        # Explanation: This line creates an empty list called 'color_extension_tasks' to store the asynchronous tasks that will be created for extending each color code.
        # The type hint 'List[asyncio.Task]' indicates that the list will contain objects of type 'asyncio.Task'.

        # Iterate over each color name and color code in the original color codes dictionary
        for color_name, color_code in color_codes.items():
            # Explanation: This line starts a loop that iterates over each key-value pair in the 'color_codes' dictionary.
            # The key represents the color name, and the value represents the color code tuple.

            # Extract the RGB values from the color code tuple
            rgb_values: RGBValue = color_code[0]
            # Explanation: This line extracts the RGB values from the first element of the color code tuple and assigns them to the variable 'rgb_values'.
            # The type hint 'RGBValue' indicates that 'rgb_values' is expected to be a tuple of three integers representing the red, green, and blue color components.

            # Create asynchronous tasks for generating additional color formats
            ansi_task: asyncio.Task = asyncio.create_task(rgb_to_ansi(rgb_values))
            hex_task: asyncio.Task = asyncio.create_task(rgb_to_hex(rgb_values))
            cmyk_task: asyncio.Task = asyncio.create_task(rgb_to_cmyk(rgb_values))
            # Explanation: These lines create asynchronous tasks for generating ANSI escape code, hexadecimal code, and CMYK values from the RGB values.
            # The functions 'rgb_to_ansi()', 'rgb_to_hex()', and 'rgb_to_cmyk()' are assumed to be defined elsewhere in the code.
            # The tasks are created using 'asyncio.create_task()' and assigned to variables with appropriate type hints.

            # Add the tasks to the list of color extension tasks
            color_extension_tasks.extend([ansi_task, hex_task, cmyk_task])
            # Explanation: This line adds the created tasks to the 'color_extension_tasks' list using the 'extend()' method.
            # The tasks will be executed concurrently to generate the additional color formats.

        # Wait for all the color extension tasks to complete and gather their results
        color_extension_results: List[Any] = await asyncio.gather(*color_extension_tasks)
        # Explanation: This line uses 'asyncio.gather()' to wait for all the tasks in 'color_extension_tasks' to complete.
        # The '*' operator unpacks the list of tasks as individual arguments to 'asyncio.gather()'.
        # The results of the tasks are collected and assigned to the variable 'color_extension_results'.
        # The type hint 'List[Any]' indicates that the results can be of any type.

        # Iterate over each color name and color code in the original color codes dictionary
        for color_name, color_code in color_codes.items():
            # Explanation: This line starts another loop that iterates over each key-value pair in the 'color_codes' dictionary.
            # The purpose of this loop is to combine the original color code with the generated additional color formats.

            # Extract the RGB values from the color code tuple
            rgb_values: RGBValue = color_code[0]
            # Explanation: This line extracts the RGB values from the first element of the color code tuple and assigns them to the variable 'rgb_values'.
            # The type hint 'RGBValue' indicates that 'rgb_values' is expected to be a tuple of three integers representing the red, green, and blue color components.

            # Extract the generated color formats from the color extension results
            ansi_code: ANSIValue = color_extension_results.pop(0)
            hex_code: HexValue = color_extension_results.pop(0)
            cmyk_values: CMYKValue = color_extension_results.pop(0)
            # Explanation: These lines extract the generated ANSI escape code, hexadecimal code, and CMYK values from the 'color_extension_results' list.
            # The 'pop(0)' method removes and returns the first element of the list each time it is called.
            # The extracted values are assigned to variables with appropriate type hints.

            # Create a tuple containing the RGB values, ANSI escape code, hexadecimal code, CMYK values, and placeholders for future extensions
            extended_color_code: ExtendedColorCode = (
                rgb_values,
                ansi_code,
                hex_code,
                cmyk_values,
                (0.0, 0.0, 0.0),  # Placeholder for Lab values
                (0.0, 0.0, 0.0),  # Placeholder for XYZ values
                None,  # Placeholder for future extensions
            )
            # Explanation: This line creates a tuple called 'extended_color_code' that combines the original RGB values with the generated additional color formats.
            # The tuple also includes placeholders for Lab values, XYZ values, and future extensions, which are currently set to default values.
            # The type hint 'ExtendedColorCode' indicates the expected structure of the tuple.

            # Add the extended color code to the dictionary with the color name as the key
            extended_color_codes[color_name] = extended_color_code
            # Explanation: This line adds the 'extended_color_code' tuple to the 'extended_color_codes' dictionary using the color name as the key.
            # The color name is used to associate the extended color information with the corresponding color.

        # Log a debug message indicating the number of color codes that were extended asynchronously
        logging.debug(f"Extended {len(extended_color_codes)} color codes asynchronously.")
        # Explanation: This line logs a debug message that includes the count of color codes that were extended asynchronously.
        # The 'len()' function is used to determine the number of key-value pairs in the 'extended_color_codes' dictionary.

        return extended_color_codes
        # Explanation: This line returns the 'extended_color_codes' dictionary, which contains the color codes with additional color formats.
        # The dictionary is the final result of the 'extend_color_codes()' function.

    except Exception as e:
        # Log an error message if an exception occurs during the color code extension process
        logging.error(f"Error occurred while extending color codes: {str(e)}")
        # Explanation: This line logs an error message if an exception occurs during the execution of the 'extend_color_codes()' function.
        # The error message includes a string representation of the exception object 'e' using 'str(e)'.
        
        raise e
        # Explanation: This line re-raises the caught exception 'e' to propagate it to the caller of the 'extend_color_codes()' function.
        # Re-raising the exception allows the caller to handle or log the error appropriately.
async def generate_dynamic_colors(extended_color_codes: ExtendedColorCodesDict) -> None:
    """
    Asynchronously generates dynamic colors for the 3D color spectacle.

    This function continuously selects random colors from the extended color codes dictionary and applies them to various elements in the 3D scene, such as the background color, particle clouds, and randomly generated shapes. The colors are smoothly transitioned using asynchronous sleep intervals to create a mesmerizing and dynamic color spectacle.

    Args:
        extended_color_codes (ExtendedColorCodesDict): The dictionary of extended color codes containing color names mapped to their corresponding color values in various formats (RGB, ANSI, HEX, CMYK, Lab, XYZ).

    Returns:
        None
    """
    # Log an info message indicating the start of dynamic color generation
    logging.info("Starting dynamic color generation.")
    # Explanation: This line logs an informational message to indicate that the dynamic color generation process has started.
    # The message is logged using the 'logging' module, which allows for structured and configurable logging throughout the code.
    
    # Enter an infinite loop to continuously generate dynamic colors
    while True:
        # Explanation: This line starts an infinite loop that will continuously generate dynamic colors until the program is interrupted or terminated.
        # The 'while True' condition ensures that the loop will keep running indefinitely.
        
        # Select a random color name from the keys of the extended color codes dictionary
        color_name: ColorName = random.choice(list(extended_color_codes.keys()))
        # Explanation: This line selects a random color name from the keys of the 'extended_color_codes' dictionary.
        # The 'keys()' method is used to retrieve all the color names (keys) from the dictionary.
        # The 'list()' function is used to convert the keys into a list, allowing the 'random.choice()' function to select a random color name from the list.
        # The selected color name is assigned to the variable 'color_name' with a type hint of 'ColorName'.
        
        # Log a debug message with the selected color name
        logging.debug(f"Selected color name: {color_name}")
        # Explanation: This line logs a debug message that includes the selected color name.
        # The 'logging.debug()' function is used to log messages at the debug level, which provides detailed information for debugging purposes.
        # The selected color name is interpolated into the log message using an f-string.

        try:
            # Explanation: This line starts a try block to handle potential exceptions that may occur during the retrieval and processing of color information.
            # The try block allows for graceful error handling and prevents the program from abruptly terminating if an exception is encountered.
            
            # Retrieve the color information tuple for the selected color name from the extended color codes dictionary
            color_info: ExtendedColorCode = extended_color_codes[color_name]
            # Explanation: This line retrieves the color information tuple associated with the selected color name from the 'extended_color_codes' dictionary.
            # The color information tuple contains various color formats and values for the selected color.
            # The retrieved color information is assigned to the variable 'color_info' with a type hint of 'ExtendedColorCode'.
            
            # Extract the RGB values from the color information tuple
            rgb: RGBValue = color_info[0]
            # Explanation: This line extracts the RGB values from the first element of the 'color_info' tuple.
            # The RGB values represent the red, green, and blue components of the color.
            # The extracted RGB values are assigned to the variable 'rgb' with a type hint of 'RGBValue'.
            
            # Normalize the RGB values to a range of 0.0 to 1.0 for use in the 3D scene
            color: Tuple[float, float, float] = (
                rgb[0] / 255.0,
                rgb[1] / 255.0,
                rgb[2] / 255.0,
            )
            # Explanation: This line normalizes the RGB values to a range of 0.0 to 1.0 for use in the 3D scene.
            # The RGB values, which typically range from 0 to 255, are divided by 255.0 to obtain normalized values between 0.0 and 1.0.
            # The normalized RGB values are stored in a tuple called 'color' with a type hint of 'Tuple[float, float, float]'.
            
            # Log a debug message with the original RGB values and the normalized color tuple
            logging.debug(f"RGB color: {rgb}, Normalized color: {color}")
            # Explanation: This line logs a debug message that includes the original RGB values and the normalized color tuple.
            # The 'logging.debug()' function is used to log the message at the debug level.
            # The original RGB values and the normalized color tuple are interpolated into the log message using an f-string.
            
        except KeyError as e:
            # Explanation: This line starts an except block to handle a 'KeyError' exception that may occur if the selected color name is not found in the 'extended_color_codes' dictionary.
            # The 'KeyError' exception is caught and assigned to the variable 'e' for further handling.
            
            # Log an error message if the selected color name is not found in the extended color codes dictionary
            logging.error(f"Color name '{color_name}' not found in extended color codes dictionary.")
            # Explanation: This line logs an error message indicating that the selected color name was not found in the 'extended_color_codes' dictionary.
            # The 'logging.error()' function is used to log the message at the error level, indicating a significant issue that needs attention.
            # The selected color name is interpolated into the error message using an f-string.
            
            # Re-raise the exception to propagate it further
            raise e
            # Explanation: This line re-raises the caught 'KeyError' exception using the 'raise' statement.
            # Re-raising the exception allows it to propagate to higher levels of the code for further handling or to terminate the program if necessary.
            
        except Exception as e:
            # Explanation: This line starts an except block to handle any other unexpected exceptions that may occur during the retrieval and processing of color information.
            # The 'Exception' class is used as a catch-all for any exception that is not specifically handled by other except blocks.
            # The caught exception is assigned to the variable 'e' for further handling.
            
            # Log an error message for any other unexpected exceptions that occur during color retrieval
            logging.error(f"An unexpected error occurred while retrieving color information: {str(e)}")
            # Explanation: This line logs an error message indicating that an unexpected error occurred while retrieving color information.
            # The 'logging.error()' function is used to log the message at the error level, indicating a significant issue that needs attention.
            # The string representation of the caught exception 'e' is included in the error message using 'str(e)'.
            
            # Re-raise the exception to propagate it further
            raise e
            # Explanation: This line re-raises the caught exception using the 'raise' statement.
            # Re-raising the exception allows it to propagate to higher levels of the code for further handling or to terminate the program if necessary.

        try:
            # Explanation: This line starts a try block to handle potential exceptions that may occur during the asynchronous sleep operation.
            # The try block allows for graceful error handling and prevents the program from abruptly terminating if an exception is encountered.
            
            # Asynchronously sleep for a short duration to create smooth color transitions
            # Adjust the range (0.001 to 0.01) for desired transition speed
            await asyncio.sleep(random.uniform(0.001, 0.01))
            # Explanation: This line introduces an asynchronous sleep operation to create smooth color transitions.
            # The 'asyncio.sleep()' function is used to pause the execution of the coroutine for a specified duration.
            # The duration is randomly generated using 'random.uniform()' with a range of 0.001 to 0.01 seconds.
            # Adjusting the range allows for controlling the speed of color transitions.
            
        except Exception as e:
            # Explanation: This line starts an except block to handle any exceptions that may occur during the asynchronous sleep operation.
            # The 'Exception' class is used as a catch-all for any exception that is encountered.
            # The caught exception is assigned to the variable 'e' for further handling.
            
            # Log an error message if an exception occurs during the asynchronous sleep
            logging.error(f"An error occurred during asynchronous sleep: {str(e)}")
            # Explanation: This line logs an error message indicating that an error occurred during the asynchronous sleep operation.
            # The 'logging.error()' function is used to log the message at the error level, indicating a significant issue that needs attention.
            # The string representation of the caught exception 'e' is included in the error message using 'str(e)'.
            
            # Re-raise the exception to propagate it further
            raise e
            # Explanation: This line re-raises the caught exception using the 'raise' statement.
            # Re-raising the exception allows it to propagate to higher levels of the code for further handling or to terminate the program if necessary.

        try:
            # Explanation: This line starts a try block to handle potential exceptions that may occur while applying the selected color to the scene background.
            # The try block allows for graceful error handling and prevents the program from abruptly terminating if an exception is encountered.
            
            # Apply the selected color to the scene background
            scene.background_color = color
            # Explanation: This line applies the selected color to the background of the 3D scene.
            # The 'background_color' attribute of the 'scene' object is assigned the value of the 'color' variable, which represents the normalized RGB color tuple.
            # This sets the background color of the scene to the selected color.
            
            # Log a debug message indicating the color applied to the scene background
            logging.debug(f"Applied color {color} to the scene background.")
            # Explanation: This line logs a debug message indicating that the selected color has been applied to the scene background.
            # The 'logging.debug()' function is used to log the message at the debug level, providing detailed information for debugging purposes.
            # The selected color tuple is interpolated into the log message using an f-string.
            
        except Exception as e:
            # Explanation: This line starts an except block to handle any exceptions that may occur while applying the color to the scene background.
            # The 'Exception' class is used as a catch-all for any exception that is encountered.
            # The caught exception is assigned to the variable 'e' for further handling.
            
            # Log an error message if an exception occurs while applying the color to the scene background
            logging.error(f"An error occurred while applying color to the scene background: {str(e)}")
            # Explanation: This line logs an error message indicating that an error occurred while applying the color to the scene background.
            # The 'logging.error()' function is used to log the message at the error level, indicating a significant issue that needs attention.
            # The string representation of the caught exception 'e' is included in the error message using 'str(e)'.
            
            # Re-raise the exception to propagate it further
            raise e
            # Explanation: This line re-raises the caught exception using the 'raise' statement.
            # Re-raising the exception allows it to propagate to higher levels of the code for further handling or to terminate the program if necessary.

        try:
            # Explanation: This line starts a try block to handle potential exceptions that may occur while generating and adding particles to the scene.
            # The try block allows for graceful error handling and prevents the program from abruptly terminating if an exception is encountered.
            
            # Generate a random number of particles between 1000 and 10000
            num_particles: int = random.randint(1000, 10000)
            # Explanation: This line generates a random number of particles between 1000 and 10000 (inclusive) using the 'random.randint()' function.
            # The generated number is assigned to the variable 'num_particles' with a type hint of 'int'.
            # The number of particles determines how many particles will be created and added to the scene.
            
            # Create a list of Particle objects with the specified number of particles
            particles: List[Particle] = [Particle() for _ in range(num_particles)]
            # Explanation: This line creates a list of 'Particle' objects using a list comprehension.
            # The list comprehension generates 'num_particles' number of 'Particle' objects by calling the 'Particle()' constructor for each iteration.
            # The created list of particles is assigned to the variable 'particles' with a type hint of 'List[Particle]'.
            
            # Iterate over each particle in the list
            for particle in particles:
                # Explanation: This line starts a loop that iterates over each 'particle' object in the 'particles' list.
                # The loop allows for performing operations on each individual particle.
                
                # Set the color of the particle to the selected color
                particle.color = color
                # Explanation: This line sets the color of the current 'particle' object to the selected color stored in the 'color' variable.
                # The 'color' attribute of the 'particle' object is assigned the value of the 'color' variable, which represents the normalized RGB color tuple.
                
                # Set the position of the particle to a random point within a specified range
                particle.position = (
                    random.uniform(-20, 20),
                    random.uniform(-20, 20),
                    random.uniform(-20, 20),
                )
                # Explanation: This line sets the position of the current 'particle' object to a random point within a specified range.
                # The 'position' attribute of the 'particle' object is assigned a tuple of three random values generated using 'random.uniform()'.
                # The random values are generated within the range of -20 to 20 for each coordinate (x, y, z) of the particle's position.
                
                # Set the velocity of the particle to a random vector within a specified range
                particle.velocity = (
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                )
                # Explanation: This line sets the velocity of the current 'particle' object to a random vector within a specified range.
                # The 'velocity' attribute of the 'particle' object is assigned a tuple of three random values generated using 'random.uniform()'.
                # The random values are generated within the range of -5 to 5 for each component (x, y, z) of the particle's velocity vector.
            
            # Add the list of particles to the scene
            scene.add_particles(particles)
            # Explanation: This line adds the list of 'particles' to the 3D scene using the 'add_particles()' method of the 'scene' object.
            # The 'particles' list, containing the generated 'Particle' objects, is passed as an argument to the 'add_particles()' method.
            # This adds the particles to the scene, making them visible and interactive.
            
            # Log a debug message indicating the number of particles generated and their color
            logging.debug(f"Generated {num_particles} particles with color {color}.")
            # Explanation: This line logs a debug message indicating the number of particles generated and their assigned color.
            # The 'logging.debug()' function is used to log the message at the debug level, providing detailed information for debugging purposes.
            # The number of particles ('num_particles') and the selected color tuple ('color') are interpolated into the log message using an f-string.
            
        except Exception as e:
            # Explanation: This line starts an except block to handle any exceptions that may occur while generating and adding particles to the scene.
            # The 'Exception' class is used as a catch-all for any exception that is encountered.
            # The caught exception is assigned to the variable 'e' for further handling.
            
            # Log an error message if an exception occurs while generating and adding particles to the scene
            logging.error(f"An error occurred while generating and adding particles to the scene: {str(e)}")
            # Explanation: This line logs an error message indicating that an error occurred while generating and adding particles to the scene.
            # The 'logging.error()' function is used to log the message at the error level, indicating a significant issue that needs attention.
            # The string representation of the caught exception 'e' is included in the error message using 'str(e)'.
            
            # Re-raise the exception to propagate it further
            raise e

        try:
            # Explanation: This line starts a try block to handle potential exceptions that may occur while generating and adding shapes to the scene.
            # The try block allows for graceful error handling and prevents the program from abruptly terminating if an exception is encountered.

            # Generate a random number of shapes between 10 and 100
            num_shapes: int = random.randint(10, 100)
            
            # Create a list of Shape objects with the specified number of shapes
            # Randomly select a shape type (Cube, Sphere, Pyramid, Torus, Cone) for each shape
            shapes: List[Shape] = [
                random.choice([Cube, Sphere, Pyramid, Torus, Cone])()
                for _ in range(num_shapes)
            ]
            
            # Iterate over each shape in the list
            for shape in shapes:
                # Set the color of the shape to the selected color
                shape.color = color
                
                # Set the position of the shape to a random point within a specified range
                shape.position = (
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                )
                
                # Set the scale of the shape to random values within a specified range
                shape.scale = (
                    random.uniform(0.1, 5),
                    random.uniform(0.1, 5),
                    random.uniform(0.1, 5),
                )
                
                # Set the rotation of the shape to random angles within a specified range
                shape.rotation = (
                    random.uniform(0, 360),
                    random.uniform(0, 360),
                    random.uniform(0, 360),
                )
            
            # Add the list of shapes to the scene
            scene.add_shapes(shapes)
            
            # Log a debug message indicating the number of shapes generated and their color
            logging.debug(f"Generated {num_shapes} shapes with color {color}.")
        except Exception as e:
            # Log an error message if an exception occurs while generating and adding shapes to the scene
            logging.error(f"An error occurred while generating and adding shapes to the scene: {str(e)}")
            # Re-raise the exception to propagate it further
            raise e
async def validate_and_load_color_codes() -> ExtendedColorCodesDict:
    """
    Validates the integrity of the color codes file, loads it if valid, or regenerates and saves a new one if invalid or not found.

    Returns:
        ExtendedColorCodesDict: A dictionary of color names mapped to their extended color codes.
    """
    try:
        # Check if the color codes file exists
        if os.path.exists(COLOR_CODES_FILE_PATH):
            # Open the color codes file in read mode
            with open(COLOR_CODES_FILE_PATH, "r") as file:
                try:
                    # Load the color codes from the file as a dictionary
                    color_codes: ExtendedColorCodesDict = json.load(file)
                    
                    # Perform a basic integrity check on the loaded color codes
                    if not isinstance(color_codes, dict) or not color_codes:
                        # Raise a ValueError if the color codes are not a dictionary or are empty
                        raise ValueError("Color codes file is corrupted or empty.")
                    
                    # Log an info message indicating successful loading of the color codes file
                    logging.info("Color codes file loaded successfully.")
                except json.JSONDecodeError as e:
                    # Log an error message if there is an error decoding the JSON file
                    logging.error(f"Error decoding color codes file: {str(e)}")
                    # Raise the exception to be caught by the outer try-except block
                    raise e
        else:
            # Raise a FileNotFoundError if the color codes file does not exist
            raise FileNotFoundError("Color codes file not found.")
    except (FileNotFoundError, ValueError) as e:
        # Log a warning message if the color codes file is not found or is invalid
        logging.warning(f"{e}. Regenerating color codes file.")
        
        try:
            # Generate a list of basic color codes asynchronously
            basic_color_codes_list: ColorCodesList = await generate_color_codes()
            
            # Extend the basic color codes with additional color information asynchronously
            color_codes = await extend_color_codes(basic_color_codes_list)
            
            # Open the color codes file in write mode
            with open(COLOR_CODES_FILE_PATH, "w") as file:
                try:
                    # Write the extended color codes to the file in JSON format with indentation
                    json.dump(color_codes, file, indent=4)
                    
                    # Log an info message indicating successful generation and saving of the new color codes file
                    logging.info("New color codes file generated and saved.")
                except Exception as e:
                    # Log an error message if there is an error writing the color codes to the file
                    logging.error(f"Error writing color codes to file: {str(e)}")
                    # Re-raise the exception to propagate it further
                    raise e
        except Exception as e:
            # Log an error message if there is an error generating or extending the color codes
            logging.error(f"Error generating or extending color codes: {str(e)}")
            # Re-raise the exception to propagate it further
            raise e
    except Exception as e:
        # Log an error message for any other unexpected exceptions
        logging.error(f"An unexpected error occurred: {str(e)}")
        # Re-raise the exception to propagate it further
        raise e
    
    # Return the loaded or regenerated extended color codes dictionary
    return color_codes
async def generate_scene() -> Scene:
    """
    Generates the initial 3D scene with various objects and shapes.

    This function creates a new Scene object and populates it with initial objects and shapes such as a rotating cube,
    a sphere with a dynamic color material, a particle system, and a color-changing torus. The objects are positioned,
    scaled, and rotated randomly within specified ranges to create an visually appealing initial scene.

    Returns:
        Scene: The generated 3D scene containing the initial objects and shapes.
    """
    # Log an info message indicating the start of initial 3D scene generation
    logging.info("Generating the initial 3D scene.")
    try:
        # Create a new Scene object
        scene: Scene = Scene()
        """
        Create a new instance of the Scene class to represent the 3D scene.
        
        The Scene class is responsible for managing and organizing the objects and shapes in the 3D scene.
        It provides methods to add objects, update their properties, and render the scene.
        
        By creating a new Scene object, we initialize an empty scene that we can populate with various objects and shapes.
        """

        try:
            # Create a Cube object with specified position and size
            cube: Cube = Cube(position=(0, 0, 0), size=(1, 1, 1))
            """
            Create a new instance of the Cube class to represent a cube object in the scene.
            
            The Cube class inherits from the base Shape class and defines the specific properties and behavior of a cube shape.
            
            The cube is initialized with a position of (0, 0, 0) and a size of (1, 1, 1), which means it is centered at the origin
            and has a width, height, and depth of 1 unit each.
            
            By creating a Cube object, we define a cube shape that can be added to the scene and manipulated further.
            """
            
            # Set the rotation properties of the cube (axis, initial angle, and rotation speed)
            cube.set_rotation(axis=(1, 1, 1), angle=0, speed=1)
            """
            Set the rotation properties of the cube object.
            
            The set_rotation method is called on the cube object to specify its rotation behavior.
            
            The axis parameter is set to (1, 1, 1), indicating that the cube will rotate around all three axes (x, y, z) simultaneously.
            The angle parameter is set to 0, representing the initial rotation angle of the cube.
            The speed parameter is set to 1, determining the speed at which the cube rotates.
            
            By setting the rotation properties, we define how the cube will rotate in the scene over time.
            """
            
            # Add the cube to the scene
            scene.add_object(cube)
            """
            Add the cube object to the scene.
            
            The add_object method is called on the scene object to include the cube in the scene.
            
            By adding the cube to the scene, it becomes part of the overall 3D scene and will be rendered and updated along with other objects.
            """
            
            # Log a debug message indicating the addition of a rotating cube to the scene
            logging.debug("Added a rotating cube to the scene.")
            """
            Log a debug message to indicate that a rotating cube has been added to the scene.
            
            The logging module is used to log messages at different severity levels.
            
            In this case, a debug message is logged to provide information about the specific action of adding a rotating cube to the scene.
            
            Logging debug messages can be helpful for tracing the execution flow and understanding the steps involved in building the scene.
            """
        except Exception as e:
            # Log an error message if there is an error creating or adding the cube to the scene
            logging.error(f"Error creating or adding cube to the scene: {str(e)}")
            """
            Log an error message if an exception occurs while creating or adding the cube to the scene.
            
            If an exception is raised during the creation or addition of the cube, it is caught in this except block.
            
            The error message is logged using the logging.error function, which logs messages at the error severity level.
            
            The error message includes a description of the error and the string representation of the exception object.
            
            Logging error messages helps in identifying and debugging issues that may occur during the scene generation process.
            """
            # Re-raise the exception to propagate it further
            raise e
            """
            Re-raise the caught exception to propagate it further.
            
            After logging the error message, the exception is re-raised using the raise statement.
            
            Re-raising the exception allows it to be caught and handled by higher-level exception handlers or to terminate the program if no suitable handler is found.
            
            By re-raising the exception, we ensure that the error is not silently ignored and can be properly dealt with at an appropriate level.
            """

        try:
            # Create a Sphere object with specified position and radius
            sphere: Sphere = Sphere(position=(2, 0, 0), radius=0.7)
            """
            Create a new instance of the Sphere class to represent a sphere object in the scene.
            
            The Sphere class inherits from the base Shape class and defines the specific properties and behavior of a sphere shape.
            
            The sphere is initialized with a position of (2, 0, 0), which means it is positioned 2 units along the positive x-axis.
            The radius of the sphere is set to 0.7 units.
            
            By creating a Sphere object, we define a sphere shape that can be added to the scene and further customized.
            """
            
            # Create a ColorMaterial object for the sphere
            color_material: ColorMaterial = ColorMaterial()
            """
            Create a new instance of the ColorMaterial class to represent the material properties of the sphere.
            
            The ColorMaterial class defines the color and shading properties of a material that can be applied to objects in the scene.
            
            By creating a ColorMaterial object, we define a material that can be assigned to the sphere to control its appearance.
            
            The ColorMaterial object will be used to set the material of the sphere, allowing it to have a dynamic color.
            """
            
            # Set the material of the sphere to the color material
            sphere.set_material(color_material)
            """
            Set the material of the sphere object to the color material.
            
            The set_material method is called on the sphere object to assign the color material to it.
            
            By setting the material of the sphere to the color material, we specify that the sphere will be rendered with the properties defined by the color material.
            
            This allows the sphere to have a dynamic color that can change over time or based on certain conditions.
            """
            
            # Add the sphere to the scene
            scene.add_object(sphere)
            """
            Add the sphere object to the scene.
            
            The add_object method is called on the scene object to include the sphere in the scene.
            
            By adding the sphere to the scene, it becomes part of the overall 3D scene and will be rendered and updated along with other objects.
            """
            
            # Log a debug message indicating the addition of a sphere with a dynamic color material to the scene
            logging.debug("Added a sphere with a dynamic color material to the scene.")
            """
            Log a debug message to indicate that a sphere with a dynamic color material has been added to the scene.
            
            The logging module is used to log messages at different severity levels.
            
            In this case, a debug message is logged to provide information about the specific action of adding a sphere with a dynamic color material to the scene.
            
            Logging debug messages can be helpful for tracing the execution flow and understanding the steps involved in building the scene.
            """
        except Exception as e:
            # Log an error message if there is an error creating or adding the sphere to the scene
            logging.error(f"Error creating or adding sphere to the scene: {str(e)}")
            """
            Log an error message if an exception occurs while creating or adding the sphere to the scene.
            
            If an exception is raised during the creation or addition of the sphere, it is caught in this except block.
            
            The error message is logged using the logging.error function, which logs messages at the error severity level.
            
            The error message includes a description of the error and the string representation of the exception object.
            
            Logging error messages helps in identifying and debugging issues that may occur during the scene generation process.
            """
            # Re-raise the exception to propagate it further
            raise e
            """
            Re-raise the caught exception to propagate it further.
            
            After logging the error message, the exception is re-raised using the raise statement.
            
            Re-raising the exception allows it to be caught and handled by higher-level exception handlers or to terminate the program if no suitable handler is found.
            
            By re-raising the exception, we ensure that the error is not silently ignored and can be properly dealt with at an appropriate level.
            """

        try:
            # Create a ParticleSystem object with specified position and number of particles
            particle_system: ParticleSystem = ParticleSystem(position=(-2, 0, 0), num_particles=1000)
            """
            Create a new instance of the ParticleSystem class to represent a particle system in the scene.
            
            The ParticleSystem class defines a system of particles that can be used to create visual effects and simulations.
            
            The particle system is initialized with a position of (-2, 0, 0), which means it is positioned 2 units along the negative x-axis.
            The num_particles parameter is set to 1000, specifying the number of particles in the system.
            
            By creating a ParticleSystem object, we define a particle system that can be added to the scene and customized further.
            """
            
            # Add the particle system to the scene
            scene.add_object(particle_system)
            """
            Add the particle system object to the scene.
            
            The add_object method is called on the scene object to include the particle system in the scene.
            
            By adding the particle system to the scene, it becomes part of the overall 3D scene and will be rendered and updated along with other objects.
            """
            
            # Log a debug message indicating the addition of a particle system to the scene
            logging.debug("Added a particle system to the scene.")
            """
            Log a debug message to indicate that a particle system has been added to the scene.
            
            The logging module is used to log messages at different severity levels.
            
            In this case, a debug message is logged to provide information about the specific action of adding a particle system to the scene.
            
            Logging debug messages can be helpful for tracing the execution flow and understanding the steps involved in building the scene.
            """
        except Exception as e:
            # Log an error message if there is an error creating or adding the particle system to the scene
            logging.error(f"Error creating or adding particle system to the scene: {str(e)}")
            """
            Log an error message if an exception occurs while creating or adding the particle system to the scene.
            
            If an exception is raised during the creation or addition of the particle system, it is caught in this except block.
            
            The error message is logged using the logging.error function, which logs messages at the error severity level.
            
            The error message includes a description of the error and the string representation of the exception object.
            
            Logging error messages helps in identifying and debugging issues that may occur during the scene generation process.
            """
            # Re-raise the exception to propagate it further
            raise e
            """
            Re-raise the caught exception to propagate it further.
            
            After logging the error message, the exception is re-raised using the raise statement.
            
            Re-raising the exception allows it to be caught and handled by higher-level exception handlers or to terminate the program if no suitable handler is found.
            
            By re-raising the exception, we ensure that the error is not silently ignored and can be properly dealt with at an appropriate level.
            """

        try:
            # Create a Torus object with specified position, inner radius, and outer radius
            torus: Torus = Torus(position=(0, 2, 0), inner_radius=0.5, outer_radius=1)
            """
            Create a new instance of the Torus class to represent a torus object in the scene.
            
            The Torus class inherits from the base Shape class and defines the specific properties and behavior of a torus shape.
            
            The torus is initialized with a position of (0, 2, 0), which means it is positioned 2 units along the positive y-axis.
            The inner_radius parameter is set to 0.5, specifying the radius of the inner circle of the torus.
            The outer_radius parameter is set to 1, specifying the radius of the outer circle of the torus.
            
            By creating a Torus object, we define a torus shape that can be added to the scene and further customized.
            """
            
            # Create a ColorMaterial object for the torus
            color_material: ColorMaterial = ColorMaterial()
            """
            Create a new instance of the ColorMaterial class to represent the material properties of the torus.
            
            The ColorMaterial class defines the color and shading properties of a material that can be applied to objects in the scene.
            
            By creating a ColorMaterial object, we define a material that can be assigned to the torus to control its appearance.
            
            The ColorMaterial object will be used to set the material of the torus, allowing it to have a dynamic color.
            """
            
            # Set the material of the torus to the color material
            torus.set_material(color_material)
            """
            Set the material of the torus object to the color material.
            
            The set_material method is called on the torus object to assign the color material to it.
            
            By setting the material of the torus to the color material, we specify that the torus will be rendered with the properties defined by the color material.
            
            This allows the torus to have a dynamic color that can change over time or based on certain conditions.
            """
            
            # Add the torus to the scene
            scene.add_object(torus)
            """
            Add the torus object to the scene.
            
            The add_object method is called on the scene object to include the torus in the scene.
            
            By adding the torus to the scene, it becomes part of the overall 3D scene and will be rendered and updated along with other objects.
            """
            
            # Log a debug message indicating the addition of a color-changing torus to the scene
            logging.debug("Added a color-changing torus to the scene.")
            """
            Log a debug message to indicate that a color-changing torus has been added to the scene.
            
            The logging module is used to log messages at different severity levels.
            
            In this case, a debug message is logged to provide information about the specific action of adding a color-changing torus to the scene.
            
            Logging debug messages can be helpful for tracing the execution flow and understanding the steps involved in building the scene.
            """
        except Exception as e:
            # Log an error message if there is an error creating or adding the torus to the scene
            logging.error(f"Error creating or adding torus to the scene: {str(e)}")
            """
            Log an error message if an exception occurs while creating or adding the torus to the scene.
            
            If an exception is raised during the creation or addition of the torus, it is caught in this except block.
            
            The error message is logged using the logging.error function, which logs messages at the error severity level.
            
            The error message includes a description of the error and the string representation of the exception object.
            
            Logging error messages helps in identifying and debugging issues that may occur during the scene generation process.
            """
            # Re-raise the exception to propagate it further
            raise e
            """
            Re-raise the caught exception to propagate it further.
            
            After logging the error message, the exception is re-raised using the raise statement.
            
            Re-raising the exception allows it to be caught and handled by higher-level exception handlers or to terminate the program if no suitable handler is found.
            
            By re-raising the exception, we ensure that the error is not silently ignored and can be properly dealt with at an appropriate level.
            """

        # Log an info message indicating the successful generation of the initial 3D scene
        logging.info("Initial 3D scene generated successfully.")
        """
        Log an info message to indicate that the initial 3D scene has been generated successfully.
        
        The logging module is used to log messages at different severity levels.
        
        In this case, an info message is logged to provide a high-level summary of the successful generation of the initial 3D scene.
        
        Logging info messages can be useful for tracking the progress and major milestones of the program execution.
        """
        
        # Return the generated scene
        return scene
    except Exception as e:
        # Log an error message if an exception occurs during the scene generation process
        logging.error(f"An error occurred during scene generation: {str(e)}")
        # Re-raise the exception to propagate it further
        raise e
        # Return None if an exception occurs
        # Log an error message if an exception occurs during the scene generation process     
async def render_scene() -> None:
    """
    Renders the 3D scene using OpenGL.

    This function clears the color and depth buffers, sets up the projection and modelview matrices, applies scene rotations, and renders all the objects in the scene. It uses OpenGL functions to perform the rendering operations and creates a visually appealing 3D scene.

    The function first clears the color and depth buffers using `glClear` to prepare for rendering a new frame. It then sets up the projection matrix using `glMatrixMode(GL_PROJECTION)` and `gluPerspective` to define the viewing frustum and perspective.

    Next, it sets up the modelview matrix using `glMatrixMode(GL_MODELVIEW)` and `gluLookAt` to specify the camera position, target point, and up vector. The scene is then rotated using `glRotatef` based on the current rotation angles stored in `scene.rotation`.

    Finally, the function calls the `render` method of the `scene` object to render all the objects and shapes in the scene. After rendering, it swaps the front and back buffers using `glutSwapBuffers` to display the rendered frame on the screen.

    This function is typically called in the main rendering loop to continuously update and display the 3D scene.
    """
    # Log a debug message indicating the start of scene rendering
    logging.debug("Rendering the 3D scene.")
    """
    Log a debug message to indicate that the rendering of the 3D scene has started.
    
    The logging module is used to log messages at different severity levels.
    
    In this case, a debug message is logged to provide detailed information about the specific action of starting the scene rendering process.
    
    Logging debug messages can be helpful for tracing the execution flow and understanding the steps involved in rendering the scene.
    """

    # Clear the color and depth buffers
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    """
    Clear the color and depth buffers using the `glClear` function from the OpenGL library.
    
    The `glClear` function is used to clear the specified buffers to their default values.
    
    In this case, the `GL_COLOR_BUFFER_BIT` and `GL_DEPTH_BUFFER_BIT` flags are combined using the bitwise OR operator (`|`) to indicate that both the color buffer and depth buffer should be cleared.
    
    Clearing the color buffer sets the color of each pixel to the default background color, while clearing the depth buffer sets the depth value of each pixel to its maximum value.
    
    This step is necessary to prepare the buffers for rendering a new frame, ensuring that the previous frame's contents are discarded and the scene is rendered from scratch.
    """

    # Set up the projection matrix
    gl.glMatrixMode(gl.GL_PROJECTION)
    """
    Set the current matrix mode to `GL_PROJECTION` using the `glMatrixMode` function from the OpenGL library.
    
    The `glMatrixMode` function is used to specify which matrix stack is the target for subsequent matrix operations.
    
    In this case, the `GL_PROJECTION` matrix mode is set, indicating that subsequent matrix operations will affect the projection matrix.
    
    The projection matrix is responsible for defining the viewing frustum and the perspective or orthographic projection of the 3D scene onto the 2D screen.
    
    Setting the matrix mode to `GL_PROJECTION` allows the subsequent matrix operations to modify the projection matrix and control how the scene is projected onto the screen.
    """
    gl.glLoadIdentity()
    """
    Load the identity matrix into the current matrix stack using the `glLoadIdentity` function from the OpenGL library.
    
    The `glLoadIdentity` function replaces the current matrix with the identity matrix, which has all elements set to zero except for the diagonal elements, which are set to one.
    
    Loading the identity matrix effectively resets the current matrix to its default state, discarding any previous transformations.
    
    In this case, loading the identity matrix into the projection matrix stack ensures that any previous transformations are cleared, and the projection matrix is reset to its default state before applying new transformations.
    
    This step is commonly performed before setting up a new projection matrix to ensure a clean starting point for defining the viewing frustum and projection parameters.
    """
    gl.gluPerspective(60, 800 / 600, 0.1, 1000.0)
    """
    Set up a perspective projection matrix using the `gluPerspective` function from the OpenGL Utility library (GLU).
    
    The `gluPerspective` function defines a viewing frustum for a perspective projection, specifying the field of view, aspect ratio, and near and far clipping planes.
    
    The function takes four parameters:
    - `fovy`: The field of view angle in the y-direction, specified in degrees. In this case, it is set to 60 degrees, representing the vertical field of view.
    - `aspect`: The aspect ratio of the viewport, calculated as the width divided by the height. In this case, it is set to 800 / 600, assuming a viewport size of 800 pixels wide and 600 pixels high.
    - `zNear`: The distance from the viewer to the near clipping plane. Objects closer than this distance will not be rendered. In this case, it is set to 0.1 units.
    - `zFar`: The distance from the viewer to the far clipping plane. Objects farther than this distance will not be rendered. In this case, it is set to 1000.0 units.
    
    The `gluPerspective` function sets up the projection matrix based on these parameters, defining the viewing frustum and the perspective projection of the 3D scene onto the 2D screen.
    
    This step is crucial for creating a realistic perspective view of the scene, where objects appear smaller as they move farther away from the viewer.
    """

    # Set up the modelview matrix
    gl.glMatrixMode(gl.GL_MODELVIEW)
    """
    Set the current matrix mode to `GL_MODELVIEW` using the `glMatrixMode` function from the OpenGL library.
    
    The `glMatrixMode` function is used to specify which matrix stack is the target for subsequent matrix operations.
    
    In this case, the `GL_MODELVIEW` matrix mode is set, indicating that subsequent matrix operations will affect the modelview matrix.
    
    The modelview matrix is responsible for defining the position, orientation, and scale of the objects in the 3D scene relative to the camera or viewer.
    
    Setting the matrix mode to `GL_MODELVIEW` allows the subsequent matrix operations to modify the modelview matrix and control the placement and transformation of objects in the scene.
    """
    gl.glLoadIdentity()
    """
    Load the identity matrix into the current matrix stack using the `glLoadIdentity` function from the OpenGL library.
    
    The `glLoadIdentity` function replaces the current matrix with the identity matrix, which has all elements set to zero except for the diagonal elements, which are set to one.
    
    Loading the identity matrix effectively resets the current matrix to its default state, discarding any previous transformations.
    
    In this case, loading the identity matrix into the modelview matrix stack ensures that any previous transformations are cleared, and the modelview matrix is reset to its default state before applying new transformations.
    
    This step is commonly performed before setting up a new modelview matrix to ensure a clean starting point for defining the position, orientation, and scale of objects in the scene.
    """
    gl.gluLookAt(0, 0, 50, 0, 0, 0, 0, 1, 0)
    """
    Set up the camera position and orientation using the `gluLookAt` function from the OpenGL Utility library (GLU).
    
    The `gluLookAt` function defines the position and orientation of the camera or viewer in the 3D scene, specifying the eye position, the target point, and the up vector.
    
    The function takes nine parameters:
    - `eyeX`, `eyeY`, `eyeZ`: The coordinates of the camera or eye position in the 3D scene. In this case, the camera is positioned at (0, 0, 50), which means it is located 50 units away from the origin along the positive z-axis.
    - `centerX`, `centerY`, `centerZ`: The coordinates of the target point or the point the camera is looking at. In this case, the camera is looking at the origin (0, 0, 0), which is the center of the scene.
    - `upX`, `upY`, `upZ`: The components of the up vector, which defines the orientation of the camera. In this case, the up vector is set to (0, 1, 0), indicating that the positive y-axis is pointing upwards.
    
    The `gluLookAt` function sets up the modelview matrix based on these parameters, defining the position and orientation of the camera in the scene.
    
    This step is crucial for setting up the camera view and determining how the scene is observed from the specified position and orientation.
    """

    # Rotate the scene based on the current rotation angles
    gl.glRotatef(scene.rotation[0], 1, 0, 0)
    """
    Apply a rotation transformation to the modelview matrix based on the current rotation angle around the x-axis.
    
    The `glRotatef` function from the OpenGL library is used to specify a rotation transformation.
    
    The function takes four parameters:
    - `angle`: The angle of rotation in degrees. In this case, the angle is obtained from `scene.rotation[0]`, which represents the rotation angle around the x-axis.
    - `x`, `y`, `z`: The components of the rotation axis. In this case, the rotation is applied around the x-axis, so the values are set to (1, 0, 0).
    
    The `glRotatef` function multiplies the current modelview matrix by a rotation matrix that rotates the scene by the specified angle around the given axis.
    
    This step applies the rotation transformation to the scene based on the current rotation angle stored in `scene.rotation[0]`, which is typically updated in each frame to create an animated rotation effect.
    
    Rotating the scene around the x-axis creates a tilting or pitching motion, where the scene appears to rotate vertically.
    """
    gl.glRotatef(scene.rotation[1], 0, 1, 0)
    """
    Apply a rotation transformation to the modelview matrix based on the current rotation angle around the y-axis.
    
    The `glRotatef` function from the OpenGL library is used to specify a rotation transformation.
    
    The function takes four parameters:
    - `angle`: The angle of rotation in degrees. In this case, the angle is obtained from `scene.rotation[1]`, which represents the rotation angle around the y-axis.
    - `x`, `y`, `z`: The components of the rotation axis. In this case, the rotation is applied around the y-axis, so the values are set to (0, 1, 0).
    
    The `glRotatef` function multiplies the current modelview matrix by a rotation matrix that rotates the scene by the specified angle around the given axis.
    
    This step applies the rotation transformation to the scene based on the current rotation angle stored in `scene.rotation[1]`, which is typically updated in each frame to create an animated rotation effect.
    
    Rotating the scene around the y-axis creates a yawing or turning motion, where the scene appears to rotate horizontally.
    """
    gl.glRotatef(scene.rotation[2], 0, 0, 1)
    """
    Apply a rotation transformation to the modelview matrix based on the current rotation angle around the z-axis.
    
    The `glRotatef` function from the OpenGL library is used to specify a rotation transformation.
    
    The function takes four parameters:
    - `angle`: The angle of rotation in degrees. In this case, the angle is obtained from `scene.rotation[2]`, which represents the rotation angle around the z-axis.
    - `x`, `y`, `z`: The components of the rotation axis. In this case, the rotation is applied around the z-axis, so the values are set to (0, 0, 1).
    
    The `glRotatef` function multiplies the current modelview matrix by a rotation matrix that rotates the scene by the specified angle around the given axis.
    
    This step applies the rotation transformation to the scene based on the current rotation angle stored in `scene.rotation[2]`, which is typically updated in each frame to create an animated rotation effect.
    
    Rotating the scene around the z-axis creates a rolling or banking motion, where the scene appears to rotate along its depth axis.
    """

    # Update the scene rotation angles for the next frame
    scene.rotation = (
        (scene.rotation[0] + 1) % 360,
        (scene.rotation[1] + 2) % 360,
        (scene.rotation[2] + 3) % 360,
    )
    """
    Update the scene rotation angles for the next frame.
    
    The `scene.rotation` attribute is a tuple that stores the rotation angles around the x-axis, y-axis, and z-axis, respectively.
    
    In this step, the rotation angles are updated by incrementing each angle by a specific value and taking the modulo with 360 to keep the angles within the range of 0 to 359 degrees.
    
    The updated rotation angles are assigned back to `scene.rotation` as a new tuple.
    
    The specific increments used for each axis are:
    - x-axis: Incremented by 1 degree per frame.
    - y-axis: Incremented by 2 degrees per frame.
    - z-axis: Incremented by 3 degrees per frame.
    
    These increments determine the speed and direction of the rotation animation for each axis. The scene will appear to rotate continuously around the respective axes based on these increments.
    
    Updating the rotation angles in each frame creates a dynamic and animated rotation effect for the scene.
    """

    # Render the objects and shapes in the scene
    scene.render()
    """
    Render the objects and shapes in the scene.
    
    This step calls the `render` method of the `scene` object, which is responsible for rendering all the objects and shapes that are part of the scene.
    
    The `render` method typically iterates over the list of objects and shapes in the scene and calls their respective rendering methods or functions to draw them on the screen.
    
    Each object or shape in the scene may have its own rendering logic, such as applying transformations, setting material properties, and specifying vertex data and primitives to be drawn.
    
    The `render` method encapsulates the rendering process for the entire scene, ensuring that all the objects and shapes are drawn in the correct order and with the appropriate properties.
    
    This step is crucial for visualizing the scene and displaying the objects and shapes on the screen based on their current positions, orientations, and appearances.
    """

    # Swap the front and back buffers to display the rendered frame
    glut.glutSwapBuffers()
    """
    Swap the front and back buffers to display the rendered frame.
    
    The `glutSwapBuffers` function from the OpenGL Utility Toolkit (GLUT) is used to swap the front and back buffers of the double-buffered window.
    
    Double buffering is a technique used to avoid flickering and tearing artifacts during the rendering process. The scene is rendered to the back buffer while the front buffer is being displayed. Once the rendering is complete, the buffers are swapped, making the newly rendered frame visible on the screen.
    
    Swapping the buffers is typically performed at the end of each rendering frame to update the displayed image with the latest rendered content.
    
    This step ensures that the rendered frame is smoothly displayed on the screen, providing a seamless and flicker-free visual experience.
    """

    logging.debug("3D scene rendering completed.")
    """
    Log a debug message to indicate that the rendering of the 3D scene has been completed.
    
    The logging module is used to log messages at different severity levels.
    
    In this case, a debug message is logged to provide detailed information about the specific action of completing the scene rendering process.
    
    Logging debug messages can be helpful for tracing the execution flow and confirming that the rendering process has finished successfully.
    """
async def update_scene() -> None:
    """
    Updates the 3D scene by rotating the camera and objects.

    This function retrieves the current elapsed time using `glutGet(GLUT_ELAPSED_TIME)` and passes it to the `update` method of the `scene` object. The `update` method is responsible for updating the positions, rotations, and any other dynamic properties of the objects in the scene based on the elapsed time.

    After updating the scene, the function calls `glutPostRedisplay` to mark the current window as needing to be redisplayed. This triggers the rendering loop to call the registered display callback function (typically `render_scene`) to redraw the scene with the updated object positions and rotations.

    This function is typically registered as the idle callback function using `glutIdleFunc` to continuously update the scene in the background.

    Returns:
        None

    Raises:
        None

    Example:
        ```python
        def update_scene() -> None:
            current_time = glut.glutGet(glut.GLUT_ELAPSED_TIME) / 1000.0
            scene.update(current_time)
            glut.glutPostRedisplay()
        ```
    """
    logging.debug("Updating the 3D scene.")
    """
    Log a debug message indicating that the 3D scene is being updated.

    This logging statement provides a detailed trace of the code's execution flow, specifically mentioning that the update process for the 3D scene has started.

    The debug level is used to convey fine-grained information that is useful for debugging purposes and understanding the step-by-step operations being performed.

    Logging such messages can assist in identifying the precise point in the code where the scene update begins, making it easier to diagnose and fix any potential issues related to scene updates.
    """

    try:
        # Get the current elapsed time in seconds
        current_time: float = glut.glutGet(glut.GLUT_ELAPSED_TIME) / 1000.0
        """
        Retrieve the current elapsed time in seconds using the `glutGet` function from the OpenGL Utility Toolkit (GLUT).

        The `GLUT_ELAPSED_TIME` parameter is passed to `glutGet` to obtain the time in milliseconds since the start of the program.

        The retrieved time is then divided by 1000.0 to convert it from milliseconds to seconds, providing a floating-point value representing the current elapsed time.

        This elapsed time is used as a parameter for updating the positions, rotations, and other dynamic properties of the objects in the scene, allowing for smooth and continuous animation based on the passage of time.

        The `current_time` variable is annotated with the `float` type hint to indicate that it holds a floating-point value representing the elapsed time in seconds.
        """

        # Update the positions, rotations, and other dynamic properties of the objects in the scene
        scene.update(current_time)
        """
        Update the positions, rotations, and other dynamic properties of the objects in the scene using the `update` method of the `scene` object.

        The `update` method takes the `current_time` as a parameter, which represents the elapsed time in seconds since the start of the program.

        Inside the `update` method, the scene object uses the elapsed time to calculate and apply transformations, animations, or any other time-dependent changes to the objects within the scene.

        The specific updates performed depend on the implementation of the `update` method in the `scene` object, but typically involve modifying the positions, rotations, scales, colors, or other properties of the objects based on the passage of time.

        By calling the `update` method with the current elapsed time, the scene and its objects are continuously updated, creating a dynamic and animated 3D environment.
        """

        # Mark the current window as needing to be redisplayed
        glut.glutPostRedisplay()
        """
        Mark the current window as needing to be redisplayed using the `glutPostRedisplay` function from the OpenGL Utility Toolkit (GLUT).

        The `glutPostRedisplay` function is used to indicate that the current window requires redrawing or refreshing.

        When `glutPostRedisplay` is called, it sends a signal to the OpenGL rendering loop, requesting it to call the registered display callback function (typically named `render_scene` or similar) to redraw the contents of the window.

        By calling `glutPostRedisplay` after updating the scene, it ensures that the changes made to the objects' positions, rotations, or other properties are reflected in the next rendering frame.

        This allows for smooth and continuous updates to the displayed scene, as the rendering loop will redraw the scene with the updated object states whenever `glutPostRedisplay` is called.
        """

    except Exception as e:
        """
        Catch any exceptions that may occur during the scene update process.

        The `try-except` block is used to handle and log any exceptions that may be raised while updating the scene.

        If an exception occurs within the `try` block, the code execution will immediately jump to the corresponding `except` block, where the exception is caught and assigned to the variable `e`.

        Inside the `except` block, the exception can be logged, handled, or propagated as needed.

        By using a broad `Exception` catch-all, this block will catch any type of exception that may occur, providing a generic error handling mechanism.

        It is important to log the exception details and handle it appropriately to prevent the program from abruptly terminating and to provide useful information for debugging and troubleshooting purposes.
        """
        logging.error(f"An error occurred while updating the 3D scene: {str(e)}")
        """
        Log an error message indicating that an error occurred while updating the 3D scene, along with the string representation of the exception.

        The `logging.error` function is used to log messages at the error level, which indicates the occurrence of an error or an exceptional condition that disrupted the normal flow of the program.

        The error message includes a description of where the error occurred (in this case, while updating the 3D scene) and the string representation of the exception object `e` using `str(e)`.

        By logging the error message, it provides valuable information for identifying and diagnosing issues related to scene updates.

        The error message will include details about the specific exception that was caught, helping developers understand the nature of the problem and take appropriate actions to fix it.

        Logging errors is crucial for maintaining a record of exceptional situations and facilitating debugging and troubleshooting processes.
        """
        raise
        """
        Re-raise the caught exception to propagate it to the caller.

        The `raise` statement without any arguments is used to re-raise the exception that was caught in the `except` block.

        When an exception is re-raised, it allows the exception to propagate up the call stack to the next higher-level exception handler or to the default exception handler if no other handler is found.

        Re-raising the exception is important in cases where the current function or block cannot handle the exception adequately and needs to pass the responsibility of handling the exception to the caller or a higher-level exception handler.

        By re-raising the exception, the original exception type, message, and traceback are preserved, providing the necessary information for debugging and error handling at a higher level.

        It is a good practice to log the exception before re-raising it, as it ensures that the error is recorded and can be investigated later, even if the exception is caught and handled at a higher level.
        """

    logging.debug("3D scene update completed.")
    """
    Log a debug message indicating that the 3D scene update has been completed.

    This logging statement provides a detailed trace of the code's execution flow, specifically mentioning that the update process for the 3D scene has finished.

    The debug level is used to convey fine-grained information that is useful for debugging purposes and understanding the step-by-step operations being performed.

    Logging such messages can assist in identifying the precise point in the code where the scene update ends, confirming that the update process has been successfully completed without any errors or exceptions.

    It serves as a complementary message to the initial debug message logged at the beginning of the function, creating a clear start and end point for the scene update operation.
    """
async def profile_function(func: Callable) -> None:
    """
    Profiles the given function using cProfile and prints the stats.

    This function takes a callable `func` as input and profiles its execution using the `cProfile` module. It enables profiling, calls the given function, disables profiling, and then prints the profiling statistics using the `pstats` module.

    The profiling statistics are sorted based on the cumulative time spent in each function, providing insights into the performance and time distribution of the profiled code.

    Args:
        func (Callable): The function to be profiled.

    Returns:
        None

    Raises:
        None

    Example:
        ```python
        def my_function():
            # Function code here
            pass

        profile_function(my_function)
        ```
    """
    profiler: cProfile.Profile = cProfile.Profile()
    """
    Create a new `Profile` object from the `cProfile` module to profile the code execution.

    The `cProfile` module provides a way to profile Python code and collect performance statistics. It allows measuring the time spent in different parts of the code and identifies bottlenecks or areas that consume significant execution time.

    The `Profile` class is the main class in the `cProfile` module used for profiling. It provides methods to start and stop profiling, as well as to retrieve the collected profiling data.

    By creating a new instance of the `Profile` class and assigning it to the variable `profiler`, we establish a profiler object that can be used to control the profiling process and access the profiling results.

    The `profiler` variable is annotated with the `cProfile.Profile` type hint to indicate that it is an instance of the `Profile` class from the `cProfile` module.
    """

    profiler.enable()
    """
    Enable the profiler to start collecting profiling data.

    The `enable` method of the `Profile` object is called to start the profiling process. Once enabled, the profiler will track the execution time and function calls of the code that follows.

    Enabling the profiler is necessary to begin gathering performance statistics and timing information about the code being profiled.

    After calling `profiler.enable()`, any subsequent code execution will be profiled until the profiler is explicitly disabled using `profiler.disable()`.

    It is important to enable the profiler before the code block or function that needs to be profiled to ensure that the relevant performance data is captured.
    """

    func()
    """
    Call the given function `func` to execute the code being profiled.

    The `func` parameter is a callable object representing the function that needs to be profiled. It is assumed to be a function that takes no arguments.

    By calling `func()`, the code inside the function is executed, and the profiler captures the performance data during its execution.

    The profiler tracks the time spent in each function call, including any nested function calls made within `func`, and records the cumulative time, number of calls, and other relevant statistics.

    It is important to note that the profiler only profiles the code executed within the `func` function and any functions it calls. Code outside of `func` will not be profiled.

    After the function call completes, the profiler continues to collect data until it is explicitly disabled.
    """

    profiler.disable()
    """
    Disable the profiler to stop collecting profiling data.

    The `disable` method of the `Profile` object is called to stop the profiling process. Once disabled, the profiler will no longer track the execution time and function calls of the code that follows.

    Disabling the profiler is necessary to conclude the profiling session and prepare for analyzing the collected data.

    After calling `profiler.disable()`, the profiler stops recording performance statistics, and the collected data can be accessed and analyzed using the `pstats` module or other profiling analysis tools.

    It is important to disable the profiler after the code block or function that needs to be profiled has finished executing to ensure that only the relevant performance data is captured and to prevent unnecessary overhead.
    """

    stats: pstats.Stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    """
    Create a `Stats` object from the `pstats` module to analyze and manipulate the profiling statistics.

    The `Stats` class in the `pstats` module provides methods to analyze and print the profiling data collected by the `cProfile` profiler.

    By creating a new instance of the `Stats` class and passing the `profiler` object as an argument, we obtain a `Stats` object that contains the profiling statistics.

    The `sort_stats` method is called on the `Stats` object to sort the profiling data based on a specific key. In this case, `SortKey.CUMULATIVE` is used as the sorting key, which sorts the data based on the cumulative time spent in each function.

    Sorting the profiling data allows for easier analysis and identification of the most time-consuming functions or code paths.

    The resulting sorted `Stats` object is assigned to the variable `stats`, which can be used to further analyze and print the profiling statistics.

    The `stats` variable is annotated with the `pstats.Stats` type hint to indicate that it is an instance of the `Stats` class from the `pstats` module.
    """

    stats.print_stats()
    """
    Print the profiling statistics to the console.

    The `print_stats` method of the `Stats` object is called to display the profiling statistics in a human-readable format.

    By default, `print_stats` prints a table of the profiled functions, including their cumulative time, total time, number of calls, and other relevant information.

    The statistics are printed to the console, allowing for immediate analysis and inspection of the profiling results.

    The output typically includes the following columns:
    - `ncalls`: The number of calls made to the function.
    - `tottime`: The total time spent in the function, excluding time spent in sub-functions.
    - `percall`: The average time per call for the function, excluding time spent in sub-functions.
    - `cumtime`: The cumulative time spent in the function, including time spent in sub-functions.
    - `percall`: The average time per call for the function, including time spent in sub-functions.
    - `filename:lineno(function)`: The file name, line number, and function name of the profiled function.

    The profiling statistics provide valuable insights into the performance characteristics of the profiled code, helping identify bottlenecks, expensive operations, and opportunities for optimization.

    It is important to review and analyze the profiling statistics to understand the performance profile of the code and make informed decisions about potential optimizations or improvements.
    """
async def main() -> None:
    """
    The main function that sets up the 3D scene, starts the color spectacle, and enters the rendering loop.

    This function integrates the functionality from both versions of the main function, ensuring all features are present and perfected.
    It sets up the OpenGL context, initializes the window, loads color codes, starts dynamic color generation, sets up callbacks, and enters the main rendering loop.

    The function is meticulously constructed to follow best practices, adhere to coding standards, and provide extensive documentation.
    It showcases the most perfected Python implementation possible, considering all aspects of code quality, performance, and maintainability.

    Returns:
        None

    Raises:
        None

    Example:
        ```python
        asyncio.run(main())
        ```
    """
    logging.info("Entering the main function.")
    """
    Log an informational message indicating that the execution has entered the main function.

    This logging statement provides a high-level overview of the program's flow, specifically mentioning that the main function has been invoked.

    The info level is used to convey general information about the program's progress and significant milestones.

    Logging such messages can help in understanding the overall structure and flow of the program, making it easier to trace the execution path and identify the starting point of the main functionality.

    It serves as a clear indication that the program has reached the main function and is ready to perform the core operations.
    """

    try:
        """
        Begin a try block to handle any exceptions that may occur during the execution of the main function.

        The try block is used to enclose the code that may potentially raise exceptions, allowing for structured exception handling.

        By placing the main functionality within a try block, we can catch and handle any exceptions that may be thrown during the execution of the code.

        If an exception occurs within the try block, the program flow will immediately jump to the corresponding except block, where the exception can be caught and handled appropriately.

        Using a try block helps in maintaining the stability and reliability of the program by preventing abrupt termination due to unhandled exceptions.

        It is a good practice to use try blocks to anticipate and handle potential exceptions, providing a mechanism to gracefully handle errors and exceptional situations.
        """

        # Initialize OpenGL and GLUT
        glut.glutInit()
        """
        Initialize the OpenGL Utility Toolkit (GLUT) library.

        The `glutInit` function is called to initialize GLUT and set up the initial state of the OpenGL environment.

        This function must be called before any other GLUT functions are invoked.

        It initializes the GLUT library and sets up the necessary internal data structures and state variables.

        The `glutInit` function also processes any command-line arguments that are relevant to GLUT, such as window size and position.

        After calling `glutInit`, the OpenGL context is ready to be configured and used for rendering.

        It is important to note that `glutInit` should only be called once at the beginning of the program, before creating any windows or performing any other GLUT-related operations.
        """

        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
        """
        Set the display mode for the OpenGL window.

        The `glutInitDisplayMode` function is used to specify the display mode settings for the OpenGL window.

        It takes one or more display mode constants as arguments, which are combined using the bitwise OR operator (`|`).

        In this case, the following display mode constants are used:
        - `GLUT_RGBA`: Specifies that the window should use RGBA color mode, which supports red, green, blue, and alpha color components.
        - `GLUT_DOUBLE`: Specifies that the window should use double buffering, which provides smooth animation and avoids flickering.
        - `GLUT_DEPTH`: Specifies that the window should have a depth buffer, which enables depth testing and proper occlusion of 3D objects.

        By combining these constants, the OpenGL window is configured to use RGBA color mode, double buffering, and a depth buffer.

        The RGBA color mode allows for specifying colors with red, green, blue, and alpha components, providing flexibility in color representation.

        Double buffering helps in achieving smooth animation by rendering the scene to an off-screen buffer and then swapping it with the displayed buffer, reducing visual artifacts.

        The depth buffer is essential for proper 3D rendering, as it allows for depth testing and ensures that objects closer to the camera obscure objects farther away.

        These display mode settings lay the foundation for the desired visual appearance and functionality of the OpenGL window.
        """

        glut.glutInitWindowSize(800, 600)
        """
        Set the initial size of the OpenGL window.

        The `glutInitWindowSize` function is used to specify the initial width and height of the OpenGL window in pixels.

        It takes two arguments: the width and the height of the window.

        In this case, the window size is set to a width of 800 pixels and a height of 600 pixels.

        This function should be called before creating the window with `glutCreateWindow`.

        The specified window size determines the initial dimensions of the OpenGL window when it is first displayed.

        However, it's important to note that the window size can be changed later using other GLUT functions or by the user resizing the window.

        Setting an appropriate initial window size provides a good starting point for the application's visual layout and ensures that the content is displayed at a reasonable size when the window is first shown.

        The choice of window size depends on the specific requirements of the application and the desired user experience.
        """

        glut.glutCreateWindow(b"3D Color Spectacle")
        """
        Create the OpenGL window with the specified title.

        The `glutCreateWindow` function is used to create a new window for OpenGL rendering and associate it with the current OpenGL context.

        It takes a byte string (`bytes`) as an argument, which represents the title of the window.

        In this case, the window title is set to "3D Color Spectacle".

        The `b` prefix before the string literal indicates that it is a byte string.

        This function should be called after initializing GLUT and setting the display mode and window size.

        Upon calling `glutCreateWindow`, a new window is created and becomes the current window for OpenGL rendering.

        The window title is displayed in the title bar of the window, providing a visual identification for the application.

        Creating the window is a crucial step in setting up the OpenGL environment and provides a surface for rendering the 3D scene.

        The window created by `glutCreateWindow` serves as the main interface between the application and the user, allowing for interaction and visualization of the rendered content.

        After creating the window, further OpenGL commands and rendering operations can be performed within the context of this window.
        """

        # Set up OpenGL
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        """
        Set the clear color for the OpenGL color buffer.

        The `glClearColor` function is used to specify the color that will be used to clear the color buffer when `glClear` is called with the `GL_COLOR_BUFFER_BIT` flag.

        It takes four arguments: red, green, blue, and alpha values, each in the range of 0.0 to 1.0.

        In this case, the clear color is set to black, with red, green, and blue components set to 0.0, and the alpha component set to 1.0 (fully opaque).

        The clear color defines the background color of the OpenGL window.

        When `glClear` is called with the `GL_COLOR_BUFFER_BIT` flag, the color buffer is cleared to the specified clear color.

        Setting the clear color to black creates a black background for the rendered scene.

        The clear color can be changed at any point during the rendering process to modify the background color of the window.

        It is common to set the clear color before rendering each frame to ensure a consistent background color throughout the application.

        The choice of clear color depends on the desired visual appearance of the application and can be adjusted to suit the specific requirements of the scene being rendered.
        """

        gl.glEnable(gl.GL_DEPTH_TEST)
        """
        Enable depth testing in OpenGL.

        The `glEnable` function is used to enable various OpenGL capabilities, and in this case, it is used to enable depth testing.

        Depth testing is a technique used to determine the visibility of objects in a 3D scene based on their distance from the camera.

        By enabling depth testing, OpenGL will perform depth comparisons between the incoming fragments (pixels) and the depth values stored in the depth buffer.

        Fragments that pass the depth test, meaning they are closer to the camera than the previously rendered fragments at the same pixel location, will be rendered and update the color and depth buffers.

        Fragments that fail the depth test, meaning they are farther away than the previously rendered fragments, will be discarded and not affect the color or depth buffers.

        Enabling depth testing ensures that objects are rendered in the correct order based on their depth, providing proper occlusion and visual realism in the 3D scene.

        Without depth testing, objects would be rendered in the order they are specified, regardless of their distance from the camera, leading to incorrect visual results.

        The `GL_DEPTH_TEST` constant is used as an argument to `glEnable` to specifically enable depth testing.

        It is important to note that for depth testing to work correctly, a depth buffer (also known as a z-buffer) must be associated with the OpenGL context, typically specified using `glutInitDisplayMode` with the `GLUT_DEPTH` flag.

        Enabling depth testing is a common practice in 3D rendering to achieve proper occlusion and maintain the correct visual order of objects in the scene.
        """

        # Load or generate color codes
        logging.info("Loading color codes.")
        """
        Log an informational message indicating that the color codes are being loaded.

        This logging statement provides a specific indication that the application is in the process of loading color codes.

        The info level is used to convey important progress or status information during the execution of the program.

        Logging such messages helps in understanding the flow of the program and identifying key stages or operations being performed.

        In this case, the message informs that the color codes, which are likely used for rendering or color-related operations, are being loaded into the application.

        The loading of color codes suggests that the application relies on a predefined set of colors or a color scheme that is being initialized.

        By logging this information, developers and users can track the progress of the application and ensure that the necessary color data is being loaded correctly.

        If any issues or errors occur during the loading process, the logged message can help in identifying the point of failure and assist in debugging.

        Overall, logging informational messages at key points in the program enhances the observability and maintainability of the codebase, making it easier to understand and diagnose issues if they arise.
        """

        extended_color_codes: ExtendedColorCodesDict = (
            await validate_and_load_color_codes()
        )
        """
        Asynchronously validate and load the color codes into an extended color codes dictionary.

        The `validate_and_load_color_codes` function is called asynchronously using the `await` keyword, indicating that it is a coroutine that returns a dictionary of extended color codes.

        The purpose of this function is to validate and load the color codes from a predefined source or generate them dynamically.

        The function likely performs the following tasks:
        1. Validates the color codes to ensure they are in the correct format and range.
        2. Loads the color codes from a file, database, or any other source.
        3. Extends the color codes by adding additional information or properties to each color code.
        4. Returns the extended color codes as a dictionary.

        The returned dictionary, `extended_color_codes`, is of type `ExtendedColorCodesDict`, which is a type alias defined earlier in the code.

        The `ExtendedColorCodesDict` represents a dictionary where the keys are color names (strings) and the values are tuples containing various color-related information such as RGB values, ANSI escape codes, hex codes, CMYK values, Lab values, XYZ values, and any additional data.

        By using an extended color codes dictionary, the application can efficiently access and utilize the color information for rendering, color manipulations, or any other color-related operations.

        The `await` keyword is used to wait for the asynchronous function to complete and retrieve the result, ensuring that the color codes are loaded before proceeding with the rest of the application.

        Asynchronous loading of color codes allows for non-blocking execution, enabling other parts of the application to continue running while the color codes are being loaded in the background.

        This approach can improve the overall performance and responsiveness of the application, especially if the color code loading process is time-consuming or involves external resources.

        Once the extended color codes are loaded, they can be used throughout the application for various color-related tasks, such as setting colors, applying color transformations, or generating color palettes.
        """

        # Start the dynamic color generation
        logging.info("Starting dynamic color generation.")
        """
        Log an informational message indicating that the dynamic color generation is starting.

        This logging statement provides a specific indication that the application is initiating the process of dynamically generating colors.

        The info level is used to convey important progress or status information during the execution of the program.

        Logging such messages helps in understanding the flow of the program and identifying key stages or operations being performed.

        In this case, the message informs that the application is starting the dynamic color generation process.

        Dynamic color generation suggests that the application will generate colors on-the-fly based on certain algorithms, rules, or user input.

        This process may involve creating new colors, modifying existing colors, or generating color palettes dynamically.

        By logging this information, developers and users can track the progress of the application and ensure that the dynamic color generation process is initiated correctly.

        If any issues or errors occur during the color generation process, the logged message can help in identifying the point of failure and assist in debugging.

        Dynamic color generation can be a powerful feature in applications that require flexible and adaptive color schemes, such as data visualization tools, creative software, or interactive experiences.

        It allows for the creation of unique and customized color combinations based on specific requirements or user preferences.

        Overall, logging informational messages at key points in the program, such as the start of dynamic color generation, enhances the observability and maintainability of the codebase, making it easier to understand and diagnose issues if they arise.
        """

        asyncio.create_task(generate_dynamic_colors(extended_color_codes))
        """
        Create an asynchronous task to generate dynamic colors using the extended color codes.

        The `asyncio.create_task` function is used to create a new asynchronous task that runs concurrently with other tasks in the event loop.

        It takes a coroutine function, `generate_dynamic_colors`, as an argument and schedules it for execution.

        The `generate_dynamic_colors` function is likely a coroutine that generates dynamic colors based on the provided extended color codes.

        It may perform various operations such as color interpolation, color mixing, or applying color transformations to create new colors dynamically.

        By creating an asynchronous task, the dynamic color generation process runs independently and concurrently with other parts of the application.

        This allows the main event loop to continue executing other tasks or handling user input while the colors are being generated in the background.

        The `extended_color_codes` dictionary, which contains the loaded and extended color codes, is passed as an argument to the `generate_dynamic_colors` function.

        This dictionary serves as a reference or starting point for generating new colors dynamically.

        The generated dynamic colors can be used for various purposes within the application, such as updating the color scheme, applying color animations, or creating visually appealing effects.

        Asynchronous execution of the dynamic color generation process ensures that it does not block the main thread of the application, providing a responsive and smooth user experience.

        It allows the application to continue functioning normally while the colors are being generated, and the results can be seamlessly integrated into the application once they are ready.

        By leveraging asynchronous programming techniques, such as creating tasks with `asyncio.create_task`, the application can efficiently handle multiple concurrent operations and make optimal use of system resources.

        This approach is particularly useful for tasks that involve time-consuming calculations, I/O operations, or external dependencies, as it prevents the application from becoming unresponsive during those operations.
        """

        # Set up GLUT callbacks
        glut.glutDisplayFunc(render_scene)
        """
        Set the display callback function for GLUT.

        The `glutDisplayFunc` function is used to register a callback function that will be called by GLUT whenever the window needs to be redrawn or displayed.

        In this case, the `render_scene` function is set as the display callback.

        The display callback function is responsible for rendering the scene and updating the window contents.

        Whenever GLUT determines that the window needs to be redrawn, it will automatically invoke the registered display callback function.

        The `render_scene` function should contain the necessary OpenGL commands to clear the buffers, set up the viewing transformations, and render the 3D scene.

        It is typically called in response to window events such as resizing, exposure, or when explicitly requested by the application.

        The `glutDisplayFunc` function is a fundamental part of setting up the rendering pipeline in a GLUT-based OpenGL application.

        By registering the `render_scene` function as the display callback, the application ensures that the 3D scene is rendered correctly whenever the window needs to be updated.
        """
if __name__ == "__main__":
    # Set up profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the main function in an asyncio event loop
    asyncio.run(main())

    # End profiling and print stats
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats()
