from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def _is_pem_section_marker(line):
    """Returns (begin:bool, end:bool, name:str)."""
    if line.startswith('-----BEGIN ') and line.endswith('-----'):
        return (True, False, line[11:-5])
    elif line.startswith('-----END ') and line.endswith('-----'):
        return (False, True, line[9:-5])
    else:
        return (False, False, '')