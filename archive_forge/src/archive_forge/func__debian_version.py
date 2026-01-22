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
def _debian_version(self) -> str:
    try:
        with open(os.path.join(self.etc_dir, 'debian_version'), encoding='ascii') as fp:
            return fp.readline().rstrip()
    except FileNotFoundError:
        return ''