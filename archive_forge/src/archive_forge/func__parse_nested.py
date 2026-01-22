import configparser
import copy
import os
import shlex
import sys
import botocore.exceptions
def _parse_nested(config_value):
    parsed = {}
    for line in config_value.splitlines():
        line = line.strip()
        if not line:
            continue
        key, value = line.split('=', 1)
        parsed[key.strip()] = value.strip()
    return parsed