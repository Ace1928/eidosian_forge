import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
@staticmethod
def _decode_internal(metadata_string):
    try:
        return JsonCoder().decode(metadata_string)
    except ValueError:
        return YamlCoder().decode(metadata_string)