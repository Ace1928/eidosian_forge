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
    rgb_values = rgb_sector_matrix[sector_index]

    # Adjust the RGB values by adding the base RGB adjustment
    adjusted_rgb_values = np.array(rgb_values) + base_rgb_adjustment

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


class ModelShowcase(ShowBase):
    """
    A class dedicated to rendering 3D models from models.py with dynamic lighting, color cycling, and camera manipulation.
    This class encapsulates the functionality required to display various 3D models with interactive features such as dynamic lighting and color cycling.
    """

    def __init__(self):
        """
        Initialize the ModelShowcase with enhanced rendering settings and tasks.
        This method sets up the initial state of the ModelShowcase, preparing it for rendering 3D models with dynamic properties.
        """
        super().__init__()
        self.model_index = 0
        self.model_names = self.retrieve_model_names()

    def initialize_lighting_components(self):
        """
        Initialize the lighting components for the 3D scene by creating and configuring various types of lights.
        This method meticulously constructs and configures point, ambient, and directional lights, each contributing uniquely to the scene's illumination.
        """
        self.point_light = PointLight("point_light")
        self.ambient_light = AmbientLight("ambient_light")
        self.directional_light = DirectionalLight("directional_light")

    def configure_point_light(self):
        """
        Configure the properties and position of the point light within the scene.
        """
        color_vector = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        color_tuple = tuple(color_vector)  # Convert numpy array to tuple
        position_vector = np.array([10, 20, 0], dtype=np.float32)
        self.point_light.setColor(color_tuple)
        point_light_node_path = self.render.attachNewNode(self.point_light)
        point_light_node_path.setPos(position_vector)
        self.render.setLight(point_light_node_path)

    def configure_ambient_light(self):
        """
        Configure the properties of the ambient light within the scene.
        """
        color_vector = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
        self.ambient_light.setColor(color_vector)
        ambient_light_node_path = self.render.attachNewNode(self.ambient_light)
        self.render.setLight(ambient_light_node_path)

    def configure_directional_light(self):
        """
        Configure the properties and orientation of the directional light within the scene.
        """
        color_vector = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
        orientation_vector = np.array([0, -60, 0], dtype=np.float32)
        self.directional_light.setColor(color_vector)
        directional_light_node_path = self.render.attachNewNode(self.directional_light)
        directional_light_node_path.setHpr(orientation_vector)
        self.render.setLight(directional_light_node_path)

    def setup_lights(self):
        """
        Master method to setup all lights by invoking individual configuration methods.
        This method ensures that all lights are configured and positioned as per the scene requirements.
        """
        self.initialize_lighting_components()
        self.configure_point_light()
        self.configure_ambient_light()
        self.configure_directional_light()

    def execute_camera_spin_task(self, task):
        """
        Execute the task to spin the camera around the scene based on the elapsed time.
        This method calculates the angular position of the camera and updates its position accordingly.
        It utilizes numpy arrays for all mathematical calculations to ensure efficiency and precision.
        """
        # Calculate the angle in degrees based on the task's elapsed time
        angle_degrees = task.time * 6.0  # 6 degrees per second rotation rate
        # Convert the angle from degrees to radians for trigonometric functions
        angle_radians = angle_degrees * (np.pi / 180.0)
        # Calculate the position vector using trigonometric functions to determine x and y, and set z to 3
        position_vector = np.array(
            [20 * np.sin(angle_radians), -20 * np.cos(angle_radians), 3],
            dtype=np.float32,
        )
        # Update the camera's position using the calculated position vector
        self.camera.setPos(*position_vector)
        # Set the camera to look at the origin of the scene
        self.camera.lookAt(np.array([0, 0, 0], dtype=np.float32))
        # Continue the task indefinitely
        return Task.cont

    def update_color_task(self, task):
        """
        Task to dynamically update the color of the models based on a hue rotation.
        This method adjusts the hue value incrementally and applies the new color to all geometric nodes in the scene.
        It ensures that all color transformations use numpy arrays and structured data management.
        """
        # Increment the hue value by 0.5 each frame, wrapping around at 360 degrees
        self.hue = (self.hue + 0.5) % 360
        # Convert the hue value to a color vector using a custom function
        color = from_hpr_to_vbase4(self.hue, 0, 0)
        # Find all geometric nodes in the scene and apply the new color
        for node in self.render.findAllMatches("**/+GeomNode"):
            # Set the color attribute of each node to the newly calculated color
            node.node().setAttrib(ColorAttrib.makeFlat(color))
        # Continue the task indefinitely
        return Task.cont

    def play(self):
        """
        Play the 3D model showcase with dynamic lighting and color cycling.
        """
        self.hue = 0
        self.setup_lights()
        self.taskMgr.add(self.spin_camera_task, "SpinCameraTask")
        self.taskMgr.add(self.update_color_task, "UpdateColorTask")
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


def main():
    """
    Main function to run the 3D model showcase.
    """
    showcase = ModelShowcase()
    showcase.play()


if __name__ == "__main__":
    main()
