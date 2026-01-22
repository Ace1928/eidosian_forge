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


# Type Aliases for enhanced clarity, type safety, and to avoid type/value/key errors
ColorName: TypeAlias = str  # Defining a type alias ColorName as str for color names
RGBValue: TypeAlias = Tuple[
    int, int, int
]  # Defining a type alias RGBValue as a tuple of three integers for RGB color values
CMYKValue: TypeAlias = Tuple[
    float, float, float, float
]  # Defining a type alias CMYKValue as a tuple of four floats for CMYK color values
LabValue: TypeAlias = Tuple[
    float, float, float
]  # Defining a type alias LabValue as a tuple of three floats for Lab color values
XYZValue: TypeAlias = Tuple[
    float, float, float
]  # Defining a type alias XYZValue as a tuple of three floats for XYZ color values
ColorCode: TypeAlias = Tuple[
    ColorName, RGBValue
]  # Defining a type alias ColorCode as a tuple of ColorName and RGBValue for color codes
ColorCodesList: TypeAlias = List[
    ColorCode
]  # Defining a type alias ColorCodesList as a list of ColorCode for a list of color codes
ColorCodesDict: TypeAlias = Dict[
    ColorName, Union[RGBValue, str]
]  # Defining a type alias ColorCodesDict as a dictionary mapping ColorName to either RGBValue or str for color codes dictionary
ANSIValue: TypeAlias = (
    str  # Defining a type alias ANSIValue as str for ANSI escape codes
)
HexValue: TypeAlias = (
    str  # Defining a type alias HexValue as str for hexadecimal color values
)
ExtendedColorCode: TypeAlias = Tuple[
    RGBValue, ANSIValue, HexValue, CMYKValue, LabValue, XYZValue, Any
]  # Defining a type alias ExtendedColorCode as a tuple of RGBValue, ANSIValue, HexValue, CMYKValue, LabValue, XYZValue, and Any for extended color codes
ExtendedColorCodesDict: TypeAlias = Dict[
    ColorName, ExtendedColorCode
]  # Defining a type alias ExtendedColorCodesDict as a dictionary mapping ColorName to ExtendedColorCode for extended color codes dictionary

# Initialize your 3D scene and objects here
try:
    scene = asyncio.run(
        generate_scene()
    )  # Initializing the 3D scene by running the generate_scene() function asynchronously
except NameError as e:
    logging.error(
        f"NameError occurred while initializing the 3D scene: {str(e)}"
    )  # Logging an error message if a NameError occurs during scene initialization
    raise  # Re-raising the exception to propagate it further

# Define the path to the color codes file
COLOR_CODES_FILE_PATH: str = (
    "/home/lloyd/EVIE/color_codes.json"  # Defining a constant variable COLOR_CODES_FILE_PATH as a string representing the path to the color codes JSON file
)


def random_color() -> RGBValue:
    """
    Generates a random RGB color value.

    Returns:
        RGBValue: A tuple of three integers representing the random RGB color value.
    """
    logging.debug(
        "Generating a random color."
    )  # Logging a debug message indicating that a random color is being generated
    r: int = random.randint(
        0, 255
    )  # Generating a random integer between 0 and 255 (inclusive) for the red component of the color
    g: int = random.randint(
        0, 255
    )  # Generating a random integer between 0 and 255 (inclusive) for the green component of the color
    b: int = random.randint(
        0, 255
    )  # Generating a random integer between 0 and 255 (inclusive) for the blue component of the color
    logging.debug(
        f"Generated random color: ({r}, {g}, {b})"
    )  # Logging a debug message with the generated random color values
    return (r, g, b)  # Returning the random RGB color value as a tuple


def generate_color_codes() -> ColorCodesList:
    """
    Generates a comprehensive list of color codes spanning a granular color spectrum.
    Each color is associated with a systematic name for easy identification and retrieval.
    Additionally, it includes the ANSI escape character, hex code, CMYK values, Lab values, XYZ values, and future extensions for each color.

    Returns:
        ColorCodesList: A list of tuples containing color names, their RGB values, ANSI escape codes, hex codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.
    """
    logging.info(
        "Generating basic color codes list."
    )  # Logging an info message indicating that the basic color codes list is being generated
    colors: ColorCodesList = [
        ("black0", (0, 0, 0))
    ]  # Initializing the colors list with the absolute black color code
    # Incrementally increasing shades of grey
    colors += [
        (f"grey{i}", (i, i, i)) for i in range(1, 256)
    ]  # Generating color codes for incrementally increasing shades of grey
    # Detailed Red to Orange spectrum
    colors += [
        (f"red{i}", (255, i, 0)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Red to Orange spectrum
    # Detailed Orange to Yellow spectrum
    colors += [
        (f"yellow{i}", (255, 255, i)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Orange to Yellow spectrum
    # Detailed Yellow to Green spectrum
    colors += [
        (f"green{i}", (255 - i, 255, 0)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Yellow to Green spectrum
    # Detailed Green to Blue spectrum
    colors += [
        (f"blue{i}", (0, 255, i)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Green to Blue spectrum
    # Detailed Blue to Indigo spectrum
    colors += [
        (f"indigo{i}", (0, 255 - i, 255)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Blue to Indigo spectrum
    # Detailed Indigo to Violet spectrum
    colors += [
        (f"violet{i}", (i, 0, 255)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Indigo to Violet spectrum
    # Detailed Violet to White spectrum
    colors += [
        (f"white{i}", (255, i, 255)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Violet to White spectrum
    colors.append(
        ("white255", (255, 255, 255))
    )  # Appending the pure white color code to the colors list
    logging.debug(
        f"Generated {len(colors)} color codes."
    )  # Logging a debug message with the total number of generated color codes
    return colors  # Returning the generated color codes list


# Asynchronous function to convert RGB to Hex
async def rgb_to_hex(rgb: RGBValue) -> HexValue:
    """
    Asynchronously converts an RGB value to its hexadecimal representation.

    This function takes an RGB value as input, which is a tuple containing three integers representing the red, green, and blue color components. It then converts the RGB value to its equivalent hexadecimal color code representation.

    The conversion is performed asynchronously, allowing for non-blocking execution and efficient utilization of system resources. The function uses string formatting to construct the hexadecimal color code by converting each RGB component to its two-digit hexadecimal representation.

    Args:
        rgb (RGBValue): A tuple containing the RGB values (red, green, blue), where each component is an integer in the range of 0 to 255.

    Returns:
        HexValue: The hexadecimal representation of the RGB color, as a string in the format "#RRGGBB".

    Example:
        >>> await rgb_to_hex((255, 128, 64))
        '#ff8040'
    """
    # Log a debug message indicating the RGB value being converted to hex asynchronously
    logging.debug(f"Converting RGB {rgb} to Hex asynchronously.")

    try:
        # Convert the RGB value to its hexadecimal representation using string formatting
        hex_value: HexValue = "#{:02x}{:02x}{:02x}".format(*rgb)
        return hex_value
    except Exception as e:
        # Log an error message if an exception occurs during the conversion process
        logging.error(f"Error occurred while converting RGB to Hex: {str(e)}")
        raise e


# Asynchronous function to generate ANSI escape code for color
async def rgb_to_ansi(rgb: RGBValue) -> ANSIValue:
    """
    Asynchronously generates the ANSI escape code for a given RGB color.

    This function takes an RGB value as input, which is a tuple containing three integers representing the red, green, and blue color components. It then generates the corresponding ANSI escape code for the RGB color.

    The ANSI escape code is a sequence of characters that can be used to control the formatting and appearance of text in compatible terminals or consoles. The generated escape code sets the foreground color of the text to the specified RGB color.

    The function constructs the ANSI escape code by combining the appropriate escape sequence ("\033[38;2;") with the RGB component values separated by semicolons.

    Args:
        rgb (RGBValue): A tuple containing the RGB values (red, green, blue), where each component is an integer in the range of 0 to 255.

    Returns:
        ANSIValue: The ANSI escape code for the RGB color, as a string.

    Example:
        >>> await rgb_to_ansi((255, 128, 64))
        '\033[38;2;255;128;64m'
    """
    # Log a debug message indicating the RGB value for which the ANSI code is being generated asynchronously
    logging.debug(f"Generating ANSI code for RGB {rgb} asynchronously.")

    try:
        # Generate the ANSI escape code for the RGB color
        ansi_code: ANSIValue = f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        return ansi_code
    except Exception as e:
        # Log an error message if an exception occurs during the ANSI code generation process
        logging.error(f"Error occurred while generating ANSI code for RGB: {str(e)}")
        raise e


# Asynchronous function to convert RGB to CMYK
async def rgb_to_cmyk(rgb: RGBValue) -> CMYKValue:
    """
    Asynchronously converts an RGB value to its corresponding CMYK representation.

    This function takes an RGB value as input, which is a tuple containing three integers representing the red, green, and blue color components. It then converts the RGB value to its equivalent CMYK (Cyan, Magenta, Yellow, Key/Black) color model representation.

    The conversion process involves normalizing the RGB values to the range of 0 to 1, calculating the key (black) component, and then determining the cyan, magenta, and yellow components based on the normalized RGB values and the key component.

    The function handles the special case of pure black (RGB: 0, 0, 0) separately to avoid division by zero.

    Args:
        rgb (RGBValue): A tuple containing the RGB values (red, green, blue), where each component is an integer in the range of 0 to 255.

    Returns:
        CMYKValue: A tuple containing the CMYK values (cyan, magenta, yellow, key), where each component is a float in the range of 0 to 1.

    Example:
        >>> await rgb_to_cmyk((255, 128, 64))
        (0.0, 0.5, 0.75, 0.0)
    """
    # Log a debug message indicating the RGB value being converted to CMYK asynchronously
    logging.debug(f"Converting RGB {rgb} to CMYK asynchronously.")

    try:
        # Normalize the RGB values to the range of 0 to 1
        r: float
        g: float
        b: float
        r, g, b = [x / 255.0 for x in rgb]

        # Calculate the key (black) component
        k: float = 1 - max(r, g, b)

        # Handle the special case of pure black (RGB: 0, 0, 0)
        if k == 1:
            return 0.0, 0.0, 0.0, 1.0

        # Calculate the cyan, magenta, and yellow components
        c: float = (1 - r - k) / (1 - k)
        m: float = (1 - g - k) / (1 - k)
        y: float = (1 - b - k) / (1 - k)

        return c, m, y, k
    except Exception as e:
        # Log an error message if an exception occurs during the RGB to CMYK conversion process
        logging.error(f"Error occurred while converting RGB to CMYK: {str(e)}")
        raise e


# Function to extend color codes with additional types and future extensions
async def extend_color_codes(
    color_codes_list: ColorCodesList,
) -> ExtendedColorCodesDict:
    """
    Asynchronously extends the basic color codes with additional color representations and placeholders for future extensions.

    This function takes a list of basic color codes as input, where each color code is represented as a tuple containing the color name and its corresponding RGB values. It then extends each color code by generating additional color representations such as ANSI escape codes, hexadecimal codes, CMYK values, and placeholders for future extensions.

    The function iterates over each color code in the input list and asynchronously calls the respective conversion functions (`rgb_to_ansi`, `rgb_to_hex`, `rgb_to_cmyk`) to generate the additional color representations. The extended color code is then stored in a dictionary with the color name as the key and a tuple containing the RGB values, ANSI escape code, hexadecimal code, CMYK values, and placeholders for future extensions as the value.

    Args:
        color_codes_list (ColorCodesList): A list of basic color codes, where each color code is a tuple containing the color name and its RGB values.

    Returns:
        ExtendedColorCodesDict: A dictionary of extended color codes, where each key is the color name and the value is a tuple containing the RGB values, ANSI escape code, hexadecimal code, CMYK values, and placeholders for future extensions.

    Example:
        >>> basic_color_codes = [("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255))]
        >>> extended_color_codes = await extend_color_codes(basic_color_codes)
        >>> extended_color_codes
        {
            "red": ((255, 0, 0), "\033[38;2;255;0;0m", "#ff0000", (0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None),
            "green": ((0, 255, 0), "\033[38;2;0;255;0m", "#00ff00", (1.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None),
            "blue": ((0, 0, 255), "\033[38;2;0;0;255m", "#0000ff", (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None)
        }
    """
    # Log an info message indicating that the basic color codes are being extended asynchronously with additional types
    logging.info("Asynchronously extending basic color codes with additional types.")

    try:
        # Initialize an empty dictionary to store the extended color codes
        extended_color_codes: ExtendedColorCodesDict = {}

        # Iterate over each color code in the input list
        for name, rgb in color_codes_list:
            # Generate the ANSI escape code for the RGB color asynchronously
            ansi: ANSIValue = await rgb_to_ansi(rgb)

            # Generate the hexadecimal code for the RGB color asynchronously
            hex_code: HexValue = await rgb_to_hex(rgb)

            # Generate the CMYK values for the RGB color asynchronously
            cmyk: CMYKValue = await rgb_to_cmyk(rgb)

            # Create a tuple containing the RGB values, ANSI escape code, hexadecimal code, CMYK values, and placeholders for future extensions
            extended_color_code: ExtendedColorCode = (
                rgb,
                ansi,
                hex_code,
                cmyk,
                (0.0, 0.0, 0.0),  # Placeholder for Lab values
                (0.0, 0.0, 0.0),  # Placeholder for XYZ values
                None,  # Placeholder for future extensions
            )

            # Add the extended color code to the dictionary with the color name as the key
            extended_color_codes[name] = extended_color_code

        # Log a debug message indicating the number of color codes that were extended asynchronously
        logging.debug(f"Extended {len(extended_color_codes)} color codes asynchronously.")

        return extended_color_codes
    except Exception as e:
        # Log an error message if an exception occurs during the color code extension process
        logging.error(f"Error occurred while extending color codes: {str(e)}")
        raise e


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
    
    # Enter an infinite loop to continuously generate dynamic colors
    while True:
        # Select a random color name from the keys of the extended color codes dictionary
        color_name: ColorName = random.choice(list(extended_color_codes.keys()))
        
        # Log a debug message with the selected color name
        logging.debug(f"Selected color name: {color_name}")

        try:
            # Retrieve the color information tuple for the selected color name from the extended color codes dictionary
            color_info: ExtendedColorCode = extended_color_codes[color_name]
            
            # Extract the RGB values from the color information tuple
            rgb: RGBValue = color_info[0]
            
            # Normalize the RGB values to a range of 0.0 to 1.0 for use in the 3D scene
            color: Tuple[float, float, float] = (
                rgb[0] / 255.0,
                rgb[1] / 255.0,
                rgb[2] / 255.0,
            )
            
            # Log a debug message with the original RGB values and the normalized color tuple
            logging.debug(f"RGB color: {rgb}, Normalized color: {color}")
        except KeyError as e:
            # Log an error message if the selected color name is not found in the extended color codes dictionary
            logging.error(f"Color name '{color_name}' not found in extended color codes dictionary.")
            # Re-raise the exception to propagate it further
            raise e
        except Exception as e:
            # Log an error message for any other unexpected exceptions that occur during color retrieval
            logging.error(f"An unexpected error occurred while retrieving color information: {str(e)}")
            # Re-raise the exception to propagate it further
            raise e

        try:
            # Asynchronously sleep for a short duration to create smooth color transitions
            # Adjust the range (0.001 to 0.01) for desired transition speed
            await asyncio.sleep(random.uniform(0.001, 0.01))
        except Exception as e:
            # Log an error message if an exception occurs during the asynchronous sleep
            logging.error(f"An error occurred during asynchronous sleep: {str(e)}")
            # Re-raise the exception to propagate it further
            raise e

        try:
            # Apply the selected color to the scene background
            scene.background_color = color
            
            # Log a debug message indicating the color applied to the scene background
            logging.debug(f"Applied color {color} to the scene background.")
        except Exception as e:
            # Log an error message if an exception occurs while applying the color to the scene background
            logging.error(f"An error occurred while applying color to the scene background: {str(e)}")
            # Re-raise the exception to propagate it further
            raise e

        try:
            # Generate a random number of particles between 1000 and 10000
            num_particles: int = random.randint(1000, 10000)
            
            # Create a list of Particle objects with the specified number of particles
            particles: List[Particle] = [Particle() for _ in range(num_particles)]
            
            # Iterate over each particle in the list
            for particle in particles:
                # Set the color of the particle to the selected color
                particle.color = color
                
                # Set the position of the particle to a random point within a specified range
                particle.position = (
                    random.uniform(-20, 20),
                    random.uniform(-20, 20),
                    random.uniform(-20, 20),
                )
                
                # Set the velocity of the particle to a random vector within a specified range
                particle.velocity = (
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                )
            
            # Add the list of particles to the scene
            scene.add_particles(particles)
            
            # Log a debug message indicating the number of particles generated and their color
            logging.debug(f"Generated {num_particles} particles with color {color}.")
        except Exception as e:
            # Log an error message if an exception occurs while generating and adding particles to the scene
            logging.error(f"An error occurred while generating and adding particles to the scene: {str(e)}")
            # Re-raise the exception to propagate it further
            raise e

        try:
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
        


def render_scene() -> None:
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


def update_scene() -> None:
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


def profile_function(func: Callable) -> None:
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
