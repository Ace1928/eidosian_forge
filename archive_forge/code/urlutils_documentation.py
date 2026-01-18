import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
Return a new URL for a path relative to this URL.

        Args:
          offset: A relative path, already urlencoded
        Returns: `URL` instance
        