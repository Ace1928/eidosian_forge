import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
class JsonCoder(beam.coders.Coder):
    """A coder to encode and decode JSON formatted data."""

    def __init__(self, indent=None):
        self._indent = indent

    def encode(self, obj):
        """Encodes a python object into a JSON string.

    Args:
      obj: python object.

    Returns:
      JSON string.
    """
        return json.dumps(obj, indent=self._indent, separators=(',', ': '))

    def decode(self, json_string):
        """Decodes a JSON string to a python object.

    Args:
      json_string: A JSON string.

    Returns:
      A python object.
    """
        return json.loads(json_string)