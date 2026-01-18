from base64 import standard_b64encode
from distutils import log
from distutils.errors import DistutilsOptionError
import os
import zipfile
import tempfile
import shutil
import itertools
import functools
import http.client
import urllib.parse
from .._importlib import metadata
from ..warnings import SetuptoolsDeprecationWarning
from .upload import upload

        Build up the MIME payload for the POST data
        