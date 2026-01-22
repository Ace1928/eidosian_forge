import sys
import logging
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task
from models import (
class LightingComponent:
    """
    A class dedicated to managing lighting components within a 3D scene.
    """

    def __init__(self, light_type: str, color: tuple, position: tuple=None, orientation: tuple=None):
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
        if self.light_type == 'point':
            self.light = PointLight('point_light')
        elif self.light_type == 'ambient':
            self.light = AmbientLight('ambient_light')
        elif self.light_type == 'directional':
            self.light = DirectionalLight('directional_light')
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