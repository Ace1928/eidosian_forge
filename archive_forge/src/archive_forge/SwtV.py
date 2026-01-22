import os
import json
import logging
import asyncio
import random
from typing import List, Tuple, Dict, Union, Any, Callable, TypeAlias, Optional
from itertools import cycle
import multiprocessing
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL
import cProfile
import pstats
from pstats import SortKey

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Type Aliases for enhanced clarity, type safety, and to avoid type/value/key errors
ColorName: TypeAlias = str
RGBValue: TypeAlias = Tuple[int, int, int]
CMYKValue: TypeAlias = Tuple[float, float, float, float]
LabValue: TypeAlias = Tuple[float, float, float]
XYZValue: TypeAlias = Tuple[float, float, float]
ColorCode: TypeAlias = Tuple[ColorName, RGBValue]
ColorCodesList: TypeAlias = List[ColorCode]
ColorCodesDict: TypeAlias = Dict[ColorName, Union[RGBValue, str]]
ANSIValue: TypeAlias = str
HexValue: TypeAlias = str
ExtendedColorCode: TypeAlias = Tuple[
    RGBValue, ANSIValue, HexValue, CMYKValue, LabValue, XYZValue, Any
]  # Any to include future extensions
ExtendedColorCodesDict: TypeAlias = Dict[ColorName, ExtendedColorCode]

# Define the path to the color codes file
COLOR_CODES_FILE_PATH: str = "/home/lloyd/EVIE/color_codes.json"

# Initialize your 3D scene and objects here
scene: Scene = Scene(generate_scene())


# Asynchronous function to convert RGB to Hex
async def rgb_to_hex(rgb: RGBValue) -> HexValue:
    """
    Asynchronously converts an RGB value to its hexadecimal representation.

    Args:
        rgb (RGBValue): A tuple containing the RGB values.

    Returns:
        HexValue: The hexadecimal representation of the color.
    """
    logging.debug(f"Converting RGB {rgb} to Hex asynchronously.")
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# Asynchronous function to generate ANSI escape code for color
async def rgb_to_ansi(rgb: RGBValue) -> ANSIValue:
    """
    Asynchronously generates the ANSI escape code for a given RGB color.

    Args:
        rgb (RGBValue): A tuple containing the RGB values.

    Returns:
        ANSIValue: The ANSI escape code for the color.
    """
    logging.debug(f"Generating ANSI code for RGB {rgb} asynchronously.")
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


# Asynchronous function to convert RGB to CMYK
async def rgb_to_cmyk(rgb: RGBValue) -> CMYKValue:
    """
    Asynchronously converts an RGB value to CMYK representation.

    Args:
        rgb (RGBValue): A tuple containing the RGB values.

    Returns:
        CMYKValue: The CMYK representation of the color.
    """
    logging.debug(f"Converting RGB {rgb} to CMYK asynchronously.")
    r: float
    g: float
    b: float
    r, g, b = [x / 255.0 for x in rgb]  # Normalize RGB values
    k: float = 1 - max(r, g, b)
    if k == 1:
        return 0.0, 0.0, 0.0, 1.0  # Pure black
    c: float = (1 - r - k) / (1 - k)
    m: float = (1 - g - k) / (1 - k)
    y: float = (1 - b - k) / (1 - k)
    return c, m, y, k


# Function to validate and load or regenerate the color codes file
async def validate_and_load_color_codes() -> ExtendedColorCodesDict:
    """
    Validates the integrity of the color codes file, loads it if valid, or regenerates and saves a new one if invalid or not found.

    Returns:
        ExtendedColorCodesDict: A dictionary of color names mapped to their extended color codes.
    """
    try:
        if os.path.exists(COLOR_CODES_FILE_PATH):
            with open(COLOR_CODES_FILE_PATH, "r") as file:
                color_codes: ExtendedColorCodesDict = json.load(file)
                # Perform a basic integrity check on the loaded color codes
                if not isinstance(color_codes, dict) or not color_codes:
                    raise ValueError("Color codes file is corrupted or empty.")
                logging.info("Color codes file loaded successfully.")
        else:
            raise FileNotFoundError("Color codes file not found.")
    except (FileNotFoundError, ValueError) as e:
        logging.warning(f"{e}. Regenerating color codes file.")
        basic_color_codes_list: ColorCodesList = await generate_color_codes()
        color_codes = await extend_color_codes(basic_color_codes_list)
        with open(COLOR_CODES_FILE_PATH, "w") as file:
            json.dump(color_codes, file, indent=4)
            logging.info("New color codes file generated and saved.")
    return color_codes


# Function to extend color codes with additional types and future extensions
async def extend_color_codes(
    color_codes_list: ColorCodesList,
) -> ExtendedColorCodesDict:
    """
    Asynchronously extends the basic color codes with ANSI escape codes, hex codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.

    Args:
        color_codes_list (ColorCodesList): The basic list of color codes.

    Returns:
        ExtendedColorCodesDict: A dictionary of color names mapped to a tuple of RGB values, ANSI escape codes, hex codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.
    """
    logging.info("Asynchronously extending basic color codes with additional types.")
    extended_color_codes: ExtendedColorCodesDict = {}
    for name, rgb in color_codes_list:
        ansi: ANSIValue = await rgb_to_ansi(rgb)
        hex_code: HexValue = await rgb_to_hex(rgb)
        cmyk: CMYKValue = await rgb_to_cmyk(rgb)
        extended_color_codes[name] = (
            rgb,
            ansi,
            hex_code,
            cmyk,
            (0.0, 0.0, 0.0),  # Placeholder for Lab values
            (0.0, 0.0, 0.0),  # Placeholder for XYZ values
            None,  # Placeholder for future extensions
        )
    logging.debug(f"Extended {len(extended_color_codes)} color codes asynchronously.")
    return extended_color_codes


async def generate_dynamic_colors(extended_color_codes: ExtendedColorCodesDict) -> None:
    """
    Asynchronously generates dynamic colors for the 3D color spectacle.

    Args:
        extended_color_codes (ExtendedColorCodesDict): The dictionary of extended color codes.
    """
    while True:
        color_name: ColorName = random.choice(list(extended_color_codes.keys()))
        color_info: ExtendedColorCode = extended_color_codes[color_name]
        rgb: RGBValue = color_info[0]
        color: Tuple[float, float, float] = (
            rgb[0] / 255.0,
            rgb[1] / 255.0,
            rgb[2] / 255.0,
        )
        await asyncio.sleep(random.uniform(0.001, 0.01))  # Adjust for desired speed
        # Apply the color to an object or the background
        scene.background_color = color

        # Generate random particle clouds
        num_particles: int = random.randint(1000, 10000)
        particles: List[Particle] = [Particle() for _ in range(num_particles)]
        for particle in particles:
            particle.color = color
            particle.position = (
                random.uniform(-20, 20),
                random.uniform(-20, 20),
                random.uniform(-20, 20),
            )
            particle.velocity = (
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(-5, 5),
            )
        scene.add_particles(particles)

        # Generate random shapes
        num_shapes: int = random.randint(10, 100)
        shapes: List[Shape] = [
            random.choice([Cube, Sphere, Pyramid, Torus, Cone])()
            for _ in range(num_shapes)
        ]
        for shape in shapes:
            shape.color = color
            shape.position = (
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10),
            )
            shape.scale = (
                random.uniform(0.1, 5),
                random.uniform(0.1, 5),
                random.uniform(0.1, 5),
            )
            shape.rotation = (
                random.uniform(0, 360),
                random.uniform(0, 360),
                random.uniform(0, 360),
            )
        scene.add_shapes(shapes)


def render_scene() -> None:
    """
    Renders the 3D scene for the color spectacle.
    """
    # Clear the color and depth buffers
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Set up the projection matrix
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.gluPerspective(60, 800 / 600, 0.1, 1000.0)

    # Set up the modelview matrix
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.gluLookAt(0, 0, 50, 0, 0, 0, 0, 1, 0)

    # Rotate the scene
    gl.glRotatef(scene.rotation[0], 1, 0, 0)
    gl.glRotatef(scene.rotation[1], 0, 1, 0)
    gl.glRotatef(scene.rotation[2], 0, 0, 1)
    scene.rotation = (
        (scene.rotation[0] + 1) % 360,
        (scene.rotation[1] + 2) % 360,
        (scene.rotation[2] + 3) % 360,
    )

    # Render the objects in the scene
    scene.render()

    # Swap the front and back buffers
    glut.glutSwapBuffers()


async def main() -> None:
    """
    Main entry point for the asynchronous color spectacle program.
    """
    # Set up your 3D library and window here, e.g., using PyOpenGL with GLUT
    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow(b"3D Color Spectacle")
    glut.glutDisplayFunc(render_scene)
    glut.glutIdleFunc(render_scene)

    extended_color_codes: ExtendedColorCodesDict = await validate_and_load_color_codes()

    # Start the dynamic color generation in the background
    asyncio.create_task(generate_dynamic_colors(extended_color_codes))

    # Enable depth testing
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Enter the main loop of the 3D library (this is blocking)
    glut.glutMainLoop()


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
