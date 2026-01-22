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
def _parse_os_release_content(lines: TextIO) -> Dict[str, str]:
    """
        Parse the lines of an os-release file.

        Parameters:

        * lines: Iterable through the lines in the os-release file.
                 Each line must be a unicode string or a UTF-8 encoded byte
                 string.

        Returns:
            A dictionary containing all information items.
        """
    props = {}
    lexer = shlex.shlex(lines, posix=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
    for token in tokens:
        if '=' in token:
            k, v = token.split('=', 1)
            props[k.lower()] = v
    if 'version' in props:
        match = re.search('\\((\\D+)\\)|,\\s*(\\D+)', props['version'])
        if match:
            release_codename = match.group(1) or match.group(2)
            props['codename'] = props['release_codename'] = release_codename
    if 'version_codename' in props:
        props['codename'] = props['version_codename']
    elif 'ubuntu_codename' in props:
        props['codename'] = props['ubuntu_codename']
    return props