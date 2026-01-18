import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
def version_parts(self, best: bool=False) -> Tuple[str, str, str]:
    """
        Return the version of the OS distribution, as a tuple of version
        numbers.

        For details, see :func:`distro.version_parts`.
        """
    version_str = self.version(best=best)
    if version_str:
        version_regex = re.compile('(\\d+)\\.?(\\d+)?\\.?(\\d+)?')
        matches = version_regex.match(version_str)
        if matches:
            major, minor, build_number = matches.groups()
            return (major, minor or '', build_number or '')
    return ('', '', '')