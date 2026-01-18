import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def validate_js_path(registered_paths, package_name, path_in_package_dist):
    if package_name not in registered_paths:
        raise exceptions.DependencyException(f'\n            Error loading dependency. "{package_name}" is not a registered library.\n            Registered libraries are:\n            {list(registered_paths.keys())}\n            ')
    if path_in_package_dist not in registered_paths[package_name]:
        raise exceptions.DependencyException(f'\n            "{package_name}" is registered but the path requested is not valid.\n            The path requested: "{path_in_package_dist}"\n            List of registered paths: {registered_paths}\n            ')