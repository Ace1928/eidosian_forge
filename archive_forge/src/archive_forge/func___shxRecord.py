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
def __shxRecord(self, offset, length):
    """Writes the shx records."""
    f = self.__getFileObj(self.shx)
    try:
        f.write(pack('>i', offset // 2))
    except error:
        raise ShapefileException('The .shp file has reached its file size limit > 4294967294 bytes (4.29 GB). To fix this, break up your file into multiple smaller ones.')
    f.write(pack('>i', length))