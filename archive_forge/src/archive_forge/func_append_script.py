import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
def append_script(self, script):
    self._resources.append_resource(script)