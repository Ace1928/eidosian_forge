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
def _get_distver_from_redhat_release(cls):
    with open('/etc/redhat-release', 'rt') as FILE:
        dist = FILE.readline().lower().strip()
        ver = ''
        for word in dist.split():
            if re.match('^[0-9\\.]+', word):
                ver = word
                break
    return (cls._map_linux_dist(dist), ver)