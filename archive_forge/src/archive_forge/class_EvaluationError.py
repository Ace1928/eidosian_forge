import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
class EvaluationError(Exception):
    """Raised if an exception occurred in safe_eval."""