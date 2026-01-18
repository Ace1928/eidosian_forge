import sys
import logging
import math
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Advanced3DCube(ShowBase):
    """
    This class encapsulates a complex 3D cube rendered using Panda3D with advanced features
    like dynamic lighting, color cycling, and camera manipulation to provide a comprehensive
    demonstration of 3D capabilities.
    """

    def __init__(self):
        """
        Initializes the Advanced3DCube with advanced rendering settings and tasks.
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
        Constructs a cube from scratch using vertex data and geom nodes.
        """
        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("cube", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")

        # Define the vertices and colors for each vertex
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

        # Define the triangles that make up the cube
        tris = GeomTriangles(Geom.UHStatic)
        for i in [0, 1, 2, 3, 4, 5, 6, 7]:  # Indices for two triangles per face
            tris.addVertices(i, (i + 1) % 4, (i + 2) % 4)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("gnode")
        node.addGeom(geom)
        nodePath = self.render.attachNewNode(node)
        nodePath.setScale(0.5)

    def setup_lights(self):
        """
        Sets up various lights in the scene.
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
        color = VBase4.fromHpr((self.hue, 1, 1))
        for node in self.render.findAllMatches("**/+GeomNode"):
            node.node().setAttrib(ColorAttrib.makeFlat(color))
        return Task.cont


app = Advanced3DCube()
app.run()
