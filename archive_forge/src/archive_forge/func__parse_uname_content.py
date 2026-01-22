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
@staticmethod
def _parse_uname_content(lines: Sequence[str]) -> Dict[str, str]:
    if not lines:
        return {}
    props = {}
    match = re.search('^([^\\s]+)\\s+([\\d\\.]+)', lines[0].strip())
    if match:
        name, version = match.groups()
        if name == 'Linux':
            return {}
        props['id'] = name.lower()
        props['name'] = name
        props['release'] = version
    return props