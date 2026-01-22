import collections
import copy
import itertools
import random
import re
import warnings
def _class_matcher(target_cls):
    """Match a node if it's an instance of the given class (PRIVATE)."""

    def match(node):
        return isinstance(node, target_cls)
    return match