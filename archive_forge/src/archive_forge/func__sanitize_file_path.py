import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _sanitize_file_path(self, cwd, file_path):
    """
        Sanitize the provided file path and ensure we always return an
        absolute path, even if relative path is passed to to this function.
        """
    if file_path[0] in ['/', '\\'] or re.match('^\\w\\:.*$', file_path):
        pass
    elif re.match('^\\w\\:.*$', cwd):
        file_path = cwd + '\\' + file_path
    else:
        file_path = pjoin(cwd, file_path)
    return file_path