from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
class HasReprMimeMeta(object):

    def _repr_mimebundle_(self, include=None, exclude=None):
        data = {'image/png': 'base64-image-data'}
        metadata = {'image/png': {'width': 5, 'height': 10}}
        return (data, metadata)