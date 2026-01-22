from __future__ import annotations
import runpy
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from ansible.executor.powershell.module_manifest import PSModuleDepFinder
from ansible.module_utils.basic import FILE_COMMON_ARGUMENTS, AnsibleModule
from ansible.module_utils.six import reraise
from ansible.module_utils.common.text.converters import to_bytes, to_text
from .utils import CaptureStd, find_executable, get_module_name_from_filename
class AnsibleModuleNotInitialized(Exception):
    pass