import collections
import copy
import itertools
import random
import re
import warnings
def _identity_matcher(target):
    """Match a node to the target object by identity (PRIVATE)."""

    def match(node):
        return node is target
    return match