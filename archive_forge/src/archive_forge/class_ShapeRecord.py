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
class ShapeRecord(object):
    """A ShapeRecord object containing a shape along with its attributes.
    Provides the GeoJSON __geo_interface__ to return a Feature dictionary."""

    def __init__(self, shape=None, record=None):
        self.shape = shape
        self.record = record

    @property
    def __geo_interface__(self):
        return {'type': 'Feature', 'properties': self.record.as_dict(date_strings=True), 'geometry': None if self.shape.shapeType == NULL else self.shape.__geo_interface__}