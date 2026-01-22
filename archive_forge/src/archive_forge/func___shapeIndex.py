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
def __shapeIndex(self, i=None):
    """Returns the offset in a .shp file for a shape based on information
        in the .shx index file."""
    shx = self.shx
    if not shx or i == None:
        return None
    if not self._offsets:
        self.__shxOffsets()
    return self._offsets[i]