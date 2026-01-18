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
def zbox(self):
    """Returns the current z extremes for the shapefile."""
    return self._zbox