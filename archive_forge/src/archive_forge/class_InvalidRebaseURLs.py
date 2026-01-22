import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
class InvalidRebaseURLs(errors.PathError):
    _fmt = 'URLs differ by more than path: %(from_)r and %(to)r'

    def __init__(self, from_, to):
        self.from_ = from_
        self.to = to
        errors.PathError.__init__(self, from_, 'URLs differ by more than path.')