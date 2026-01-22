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
def _sanitize_cwd(self, cwd):
    if re.match('^\\/\\w\\:.*$', str(cwd)):
        cwd = str(cwd[1:])
    return cwd