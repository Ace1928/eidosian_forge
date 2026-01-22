from __future__ import absolute_import
import json
import six
from googleapiclient import _helpers as util
class InvalidJsonError(Error):
    """The JSON returned could not be parsed."""
    pass