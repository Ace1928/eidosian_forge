import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def DICT(self, node):
    keys = [convert_to_value(key) for key in node.keys]
    key_counts = counter(keys)
    duplicate_keys = [key for key, count in key_counts.items() if count > 1]
    for key in duplicate_keys:
        key_indices = [i for i, i_key in enumerate(keys) if i_key == key]
        values = counter((convert_to_value(node.values[index]) for index in key_indices))
        if any((count == 1 for value, count in values.items())):
            for key_index in key_indices:
                key_node = node.keys[key_index]
                if isinstance(key, VariableKey):
                    self.report(messages.MultiValueRepeatedKeyVariable, key_node, key.name)
                else:
                    self.report(messages.MultiValueRepeatedKeyLiteral, key_node, key)
    self.handleChildren(node)