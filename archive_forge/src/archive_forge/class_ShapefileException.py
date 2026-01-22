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
class ShapefileException(Exception):
    """An exception to handle shapefile specific problems."""
    pass