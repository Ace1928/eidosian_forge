from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
class StopParsing(Exception):
    """Exception raised by a TreeBuilder if it's unable to continue parsing."""
    pass