import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class StateCorrection(Exception):
    """
    Raise from within a transition method to switch to another state.

    Raise with one or two arguments: new state name, and an optional new
    transition name.
    """