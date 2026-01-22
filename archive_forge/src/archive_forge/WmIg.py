import sys
import logging
import math
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task

# Configure logging with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def hue_to_rgb(hue):
    """
    Convert a hue value (in degrees) to an RGB color with full saturation and brightness.
    """
    hue = hue % 360  # Normalize hue to be within 0-360 degrees
    c = 1.0  # Saturation and brightness are both 1.0
    x = 1 - abs((hue / 60.0 % 2) - 1)
    m = 0.0  # No adjustment needed for RGB range

    # Calculate RGB based on hue sector
    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    # Convert to RGB by adding m
    r, g, b = (r + m), (g + m), (b + m)
    return r, g, b


def from_hpr_to_vbase4(hue, pitch, roll):
    """
    Convert HPR (Hue, Pitch, Roll) to a VBase4 object, using hue for color.
    """
    r, g, b = hue_to_rgb(hue)
    alpha = 1.0  # Full opacity
    return VBase4(r, g, b, alpha)


class ModelShowcase(ShowBase):
    """
    A class to render 3D models from models.py with dynamic lighting, color cycling, and camera manipulation.
    """

    def __init__(self):
        """
        Initialize the Advanced3DCube with enhanced rendering settings and tasks.
        """
        super().__init__()

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

    def setup_lights(self):
        """
        Set up various lights in the scene.
        """
        # Point light
        pl = PointLight("pl")
        pl.setColor((1, 1, 1, 1))
        plnp = self.render.attachNewNode(pl)
        plnp.setPos(10, 20, 0)
        self.render.setLight(plnp)

        # Ambient light
        al = AmbientLight("al")
        al.setColor((0.5, 0.5, 0.5, 1))
        alnp = self.render.attachNewNode(al)
        self.render.setLight(alnp)

        # Directional light
        dl = DirectionalLight("dl")
        dl.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dl)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

    def spinCameraTask(self, task):
        """
        Task to spin the camera around the scene.
        """
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (math.pi / 180.0)
        self.camera.setPos(20 * math.sin(angleRadians), -20 * math.cos(angleRadians), 3)
        self.camera.lookAt(0, 0, 0)
        return Task.cont

    def updateColor(self, task):
        """
        Task to dynamically update the color of the cube.
        """
        self.hue = (self.hue + 0.5) % 360
        color = from_hpr_to_vbase4(self.hue, 0, 0)
        for node in self.render.findAllMatches("**/+GeomNode"):
            node.node().setAttrib(ColorAttrib.makeFlat(color))
        return Task.cont

    def play(self):
        """
        Play the 3D model showcase with dynamic lighting and color cycling.
        """
        self.hue = 0
        self.setup_lights()
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.updateColor, "UpdateColorTask")
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

    def previous_model(self):
        """
        Load the previous 3D model in the showcase.
        """

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

    def list_models(self):
        """
        List the available 3D models in the showcase.
        """
        model_names = [name[10:] for name in dir(self) if name.startswith("construct_")]
        return model_names

    def run(self):
        """
        Run the Panda3D application.
        """
        self.run()


def main():
    """
    Main function to run the 3D model showcase.
    """
    showcase = ModelShowcase()
    showcase.play()


if __name__ == "__main__":
    main()
