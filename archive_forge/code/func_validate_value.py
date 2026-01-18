import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def validate_value(self, value):
    """Validate a value name"""
    if value not in self.registry:
        raise BadOptionValue(self.name, value)