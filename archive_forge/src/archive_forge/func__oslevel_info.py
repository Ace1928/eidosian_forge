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
@cached_property
def _oslevel_info(self) -> str:
    if not self.include_oslevel:
        return ''
    try:
        stdout = subprocess.check_output('oslevel', stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ''
    return self._to_str(stdout).strip()