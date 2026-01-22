import sys
import logging
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task
import numpy as np
from models import (
    construct_triangle_sheet_with_vertex_data,
    construct_square_sheet_with_vertex_data,
    construct_circle_sheet_with_vertex_data,
    construct_cube,
    construct_sphere,
    construct_cylinder,
    construct_cone,
    construct_dodecahedron,
    construct_icosahedron,
    construct_octahedron,
    construct_tetrahedron,
    construct_conical_frustum,
    construct_cylindrical_frustum,
    construct_spherical_frustum,
    construct_torus_knot,
    construct_trefoil_knot,
    construct_mobius_strip,
    construct_klein_bottle,
    construct_torus,
)

# Configure logging with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_hue_to_rgb_vector(hue_angle_degrees):
    """
    Convert a hue value (in degrees) to an RGB color vector with full saturation and brightness.
    This function meticulously calculates the RGB values based on the hue angle provided, ensuring
    the output is a tuple of RGB values, each component ranging from 0 to 1.

    Parameters:
    - hue_angle_degrees (float): The hue angle in degrees, which will be normalized to a range of 0-360.

    Returns:
    - tuple: A tuple representing the RGB color (r, g, b), each component as a float from 0 to 1.
    """
    # Normalize the hue angle to ensure it is within the range of 0 to 360 degrees
    normalized_hue_angle = hue_angle_degrees % 360

    # Define constants for maximum saturation and brightness
    maximum_saturation = 1.0
    maximum_brightness = 1.0

    # Calculate the intermediate value 'x' used in the RGB conversion process
    intermediate_x = 1 - abs((normalized_hue_angle / 60.0 % 2) - 1)

    # Define the base adjustment for RGB values, which is zero in this case as no adjustment is needed
    base_rgb_adjustment = 0.0

    # Define the RGB sectors based on the hue angle using a structured array approach
    rgb_sector_matrix = np.array(
        [
            (maximum_saturation, intermediate_x, 0),
            (intermediate_x, maximum_saturation, 0),
            (0, maximum_saturation, intermediate_x),
            (0, intermediate_x, maximum_saturation),
            (intermediate_x, 0, maximum_saturation),
            (maximum_saturation, 0, intermediate_x),
        ],
        dtype=np.float32,
    )

    # Determine the sector index based on the normalized hue angle
    sector_index = int(normalized_hue_angle // 60)
    # Extract the RGB values from the sector matrix based on the calculated sector index
    # Ensuring that the extraction is done correctly by converting the sector index to an integer if not already
    sector_index_int = int(sector_index)
    # Utilizing advanced numpy techniques to ensure the correct data type and handling
    rgb_values = np.take(rgb_sector_matrix, sector_index_int, axis=0)

    # Adjust the RGB values by adding the base RGB adjustment
    # Using numpy's broadcasting feature to add the base adjustment to each element in the RGB values array
    adjusted_rgb_values = np.add(rgb_values, base_rgb_adjustment)
    # Return the RGB values as a tuple
    return tuple(adjusted_rgb_values)


def convert_hpr_to_vbase4_with_full_opacity(hue, pitch, roll):
    """
    Convert Hue, Pitch, and Roll (HPR) values to a VBase4 object, utilizing the hue component to determine the color.
    This function meticulously transforms the hue component into an RGB color vector, then combines it with a full opacity
    value to create a VBase4 object, which is used extensively in 3D graphics for representing color and transparency.

    Parameters:
    - hue (float): The hue component used for color conversion, representing the angle in degrees on a color wheel.
    - pitch (float): The pitch component, which is not utilized in this function but is included for complete HPR representation.
    - roll (float): The roll component, which is not utilized in this function but is included for complete HPR representation.

    Returns:
    - VBase4: An object representing the RGBA color, where RGB is derived from the hue and A (alpha) is set to 1.0 for full opacity.
    """
    # Utilize the previously defined function to convert hue to an RGB color vector
    rgb_color_vector = convert_hue_to_rgb_vector(hue)

    # Define the alpha value for full opacity
    alpha_value = 1.0

    # Create a VBase4 object with the RGB color vector and full opacity
    color_with_opacity = VBase4(*rgb_color_vector, alpha_value)

    return color_with_opacity


import numpy as np
from panda3d.core import PointLight, AmbientLight, DirectionalLight, ColorAttrib, VBase4
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task
import logging

# Configure logging with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LightingComponent:
    """
    A class dedicated to managing lighting components within a 3D scene.
    """

    def __init__(
        self,
        light_type: str,
        color: tuple,
        position: tuple = None,
        orientation: tuple = None,
    ):
        """
        Initialize a lighting component with specified type, color, position, and orientation.
        """
        self.light_type = light_type
        self.color = color
        self.position = position
        self.orientation = orientation
        self.light = None
        self.node_path = None

    def create_light(self):
        """
        Create the light based on its type and set its color.
        """
        if self.light_type == "point":
            self.light = PointLight("point_light")
        elif self.light_type == "ambient":
            self.light = AmbientLight("ambient_light")
        elif self.light_type == "directional":
            self.light = DirectionalLight("directional_light")
        self.light.setColor(self.color)

    def attach_to_render(self, render):
        """
        Attach the light to the render and set its position or orientation if applicable.
        """
        self.node_path = render.attachNewNode(self.light)
        if self.position:
            self.node_path.setPos(self.position)
        if self.orientation:
            self.node_path.setHpr(self.orientation)
        render.setLight(self.node_path)


class CameraController:
    """
    A class dedicated to controlling the camera within a 3D scene.
    """

    def __init__(self, camera):
        """
        Initialize the CameraController with a camera object.
        """
        self.camera = camera

    def spin_camera(self, task):
        """
        Spin the camera around the scene based on the elapsed time.
        """
        angle_degrees = task.time * 6.0  # 6 degrees per second rotation rate
        angle_radians = angle_degrees * (np.pi / 180.0)
        position = (20 * np.sin(angle_radians), -20 * np.cos(angle_radians), 3)
        self.camera.setPos(position)
        self.camera.lookAt((0, 0, 0))
        return Task.cont


class ColorCycler:
    """
    A class dedicated to cycling the color of models in a 3D scene.
    """

    def __init__(self, render, initial_hue=0):
        """
        Initialize the ColorCycler with a render node and an initial hue value.
        """
        self.render = render
        self.hue = initial_hue

    def update_color(self, task):
        """
        Update the color of the models based on a hue rotation.
        """
        self.hue = (self.hue + 0.5) % 360
        color = convert_hpr_to_vbase4_with_full_opacity(self.hue, 0, 0)
        for node in self.render.findAllMatches("**/+GeomNode"):
            node.node().setAttrib(ColorAttrib.makeFlat(color))
        return Task.cont


class ModelShowcase(ShowBase):
    """
    A class dedicated to rendering 3D models with dynamic lighting, color cycling, and camera manipulation.
    """

    def __init__(self):
        """
        Initialize the ModelShowcase with enhanced rendering settings and tasks.
        """
        super().__init__()
        self.model_index = 0
        self.model_names = self.retrieve_model_names()
        self.setup_lights()
        self.camera_controller = CameraController(self.camera)
        self.color_cycler = ColorCycler(self.render)

    def setup_lights(self):
        """
        Setup all lights by creating and attaching them to the render.
        """
        colors = [(1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.8, 0.8, 0.8, 1.0)]
        positions = [(10, 20, 0), None, None]
        orientations = [None, None, (0, -60, 0)]
        types = ["point", "ambient", "directional"]
        for color, position, orientation, light_type in zip(
            colors, positions, orientations, types
        ):
            light = LightingComponent(light_type, color, position, orientation)
            light.create_light()
            light.attach_to_render(self.render)

    def play(self):
        """
        Play the 3D model showcase with dynamic lighting and color cycling.
        """
        self.taskMgr.add(self.camera_controller.spin_camera, "SpinCameraTask")
        self.taskMgr.add(self.color_cycler.update_color, "UpdateColorTask")
        self.run()

    def pause(self):
        """
        Pause the 3D model showcase.
        """
        self.taskMgr.remove("SpinCameraTask")
        self.taskMgr.remove("UpdateColorTask")

    def restart(self):
        """
        Restart the 3D model showcase.
        """
        self.pause()
        self.play()

    def next_model(self):
        """
        Load the next 3D model in the showcase.
        """
        self.model_index = (self.model_index + 1) % len(self.model_names)
        self.load_model(self.model_names[self.model_index])

    def previous_model(self):
        """
        Load the previous 3D model in the showcase.
        """
        self.model_index = (self.model_index - 1) % len(self.model_names)
        self.load_model(self.model_names[self.model_index])

    def load_model(self, model_name):
        """
        Load a specific 3D model in the showcase.
        """
        model = getattr(self, f"construct_{model_name}")()
        model.reparentTo(self.render)

    def clear_scene(self):
        """
        Clear the scene of all 3D models.
        """
        self.render.removeNode()

    def retrieve_model_names(self):
        """
        Retrieve the available 3D models in the showcase.
        """
        model_names = [name[10:] for name in dir(self) if name.startswith("construct_")]
        return model_names

    def run(self):
        """
        Run the Panda3D application.
        """
        self.taskMgr.run()


if __name__ == "__main__":
    ModelShowcase().run()
