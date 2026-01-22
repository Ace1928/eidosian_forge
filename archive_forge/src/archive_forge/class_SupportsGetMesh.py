from __future__ import annotations
import functools
import operator
import re
from typing import Protocol, Sequence, cast
from . import ExifTags, Image, ImagePalette
class SupportsGetMesh(Protocol):
    """
    An object that supports the ``getmesh`` method, taking an image as an
    argument, and returning a list of tuples. Each tuple contains two tuples,
    the source box as a tuple of 4 integers, and a tuple of 8 integers for the
    final quadrilateral, in order of top left, bottom left, bottom right, top
    right.
    """

    def getmesh(self, image: Image.Image) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int, int, int, int, int]]]:
        ...