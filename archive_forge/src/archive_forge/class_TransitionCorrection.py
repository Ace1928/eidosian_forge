import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class TransitionCorrection(Exception):
    """
    Raise from within a transition method to switch to another transition.

    Raise with one argument, the new transition name.
    """