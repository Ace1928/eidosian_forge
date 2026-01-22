import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
@classmethod
def _get_distver_from_os_release(cls):
    dist = ''
    ver = ''
    with open('/etc/os-release', 'rt') as FILE:
        for line in FILE:
            line = line.strip()
            if not line:
                continue
            key, val = line.lower().split('=')
            if val[0] == val[-1] and val[0] in '"\'':
                val = val[1:-1]
            if key == 'id':
                dist = val
            elif key == 'version_id':
                ver = val
    return (cls._map_linux_dist(dist), ver)