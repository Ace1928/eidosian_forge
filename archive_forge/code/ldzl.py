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


class Advanced3DCube(ShowBase):
    """
    A class to render a 3D cube with dynamic lighting, color cycling, and camera manipulation.
    """

    def __init__(self):
        """
        Initialize the Advanced3DCube with enhanced rendering settings and tasks.
        """
        super().__init__()
        self.disableMouse()
        self.construct_cube()
        self.setup_lights()
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.hue = 0
        self.taskMgr.add(self.updateColor, "UpdateColor")

    def construct_cube(self):
        """
        Construct a cube using vertex data and geom nodes.
        """
        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("cube", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")

        # Define vertices and colors
        vertices = [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ]
        colors = [
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 0, 1, 1),
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (0, 0, 0, 1),
        ]
        for vert, col in zip(vertices, colors):
            vertex.addData3(*vert)
            color.addData4(*col)

        # Define triangles for the cube (corrected indices)
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(0, 8, 4):  # Corrected indices for each face
            tris.addVertices(i, i + 1, i + 2)
            tris.addVertices(i, i + 2, i + 3)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("gnode")
        node.addGeom(geom)
        nodePath = self.render.attachNewNode(node)
        nodePath.setScale(0.5)

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


app = Advanced3DCube()
app.run()
