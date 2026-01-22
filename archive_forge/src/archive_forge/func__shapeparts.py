from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def _shapeparts(self, parts, shapeType):
    """Internal method for adding a shape that has multiple collections of points (parts):
        lines, polygons, and multipoint shapes.
        """
    polyShape = Shape(shapeType)
    polyShape.parts = []
    polyShape.points = []
    if shapeType in (5, 15, 25, 31):
        for part in parts:
            if part[0] != part[-1]:
                part.append(part[0])
    for part in parts:
        polyShape.parts.append(len(polyShape.points))
        for point in part:
            if not isinstance(point, list):
                point = list(point)
            polyShape.points.append(point)
    self.shape(polyShape)