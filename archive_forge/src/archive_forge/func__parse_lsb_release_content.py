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
def _parse_lsb_release_content(lines: Iterable[str]) -> Dict[str, str]:
    """
        Parse the output of the lsb_release command.

        Parameters:

        * lines: Iterable through the lines of the lsb_release output.
                 Each line must be a unicode string or a UTF-8 encoded byte
                 string.

        Returns:
            A dictionary containing all information items.
        """
    props = {}
    for line in lines:
        kv = line.strip('\n').split(':', 1)
        if len(kv) != 2:
            continue
        k, v = kv
        props.update({k.replace(' ', '_').lower(): v.strip()})
    return props