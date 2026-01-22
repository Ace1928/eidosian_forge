import collections
import copy
import itertools
import random
import re
import warnings
def _function_matcher(matcher_func):
    """Safer attribute lookup -- returns False instead of raising an error (PRIVATE)."""

    def match(node):
        try:
            return matcher_func(node)
        except (LookupError, AttributeError, ValueError, TypeError):
            return False
    return match