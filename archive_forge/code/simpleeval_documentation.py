import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
Safely evaluates the expression having an attributed accessed