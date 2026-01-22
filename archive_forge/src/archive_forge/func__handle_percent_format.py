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
def _handle_percent_format(self, node):
    try:
        placeholders = parse_percent_format(node.left.value)
    except ValueError:
        self.report(messages.PercentFormatInvalidFormat, node, 'incomplete format')
        return
    named = set()
    positional_count = 0
    positional = None
    for _, placeholder in placeholders:
        if placeholder is None:
            continue
        name, _, width, precision, conversion = placeholder
        if conversion == '%':
            continue
        if conversion not in VALID_CONVERSIONS:
            self.report(messages.PercentFormatUnsupportedFormatCharacter, node, conversion)
        if positional is None and conversion:
            positional = name is None
        for part in (width, precision):
            if part is not None and '*' in part:
                if not positional:
                    self.report(messages.PercentFormatStarRequiresSequence, node)
                else:
                    positional_count += 1
        if positional and name is not None:
            self.report(messages.PercentFormatMixedPositionalAndNamed, node)
            return
        elif not positional and name is None:
            self.report(messages.PercentFormatMixedPositionalAndNamed, node)
            return
        if positional:
            positional_count += 1
        else:
            named.add(name)
    if isinstance(node.right, (ast.List, ast.Tuple)) and (not any((isinstance(elt, ast.Starred) for elt in node.right.elts))):
        substitution_count = len(node.right.elts)
        if positional and positional_count != substitution_count:
            self.report(messages.PercentFormatPositionalCountMismatch, node, positional_count, substitution_count)
        elif not positional:
            self.report(messages.PercentFormatExpectedMapping, node)
    if isinstance(node.right, ast.Dict) and all((isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.right.keys)):
        if positional and positional_count > 1:
            self.report(messages.PercentFormatExpectedSequence, node)
            return
        substitution_keys = {k.value for k in node.right.keys}
        extra_keys = substitution_keys - named
        missing_keys = named - substitution_keys
        if not positional and extra_keys:
            self.report(messages.PercentFormatExtraNamedArguments, node, ', '.join(sorted(extra_keys)))
        if not positional and missing_keys:
            self.report(messages.PercentFormatMissingArgument, node, ', '.join(sorted(missing_keys)))