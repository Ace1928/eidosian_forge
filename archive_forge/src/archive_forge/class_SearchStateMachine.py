import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class SearchStateMachine(_SearchOverride, StateMachine):
    """`StateMachine` which uses `re.search()` instead of `re.match()`."""
    pass