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
def _handle_string_dot_format(self, node):
    try:
        placeholders = tuple(parse_format_string(node.func.value.value))
    except ValueError as e:
        self.report(messages.StringDotFormatInvalidFormat, node, e)
        return
    auto = None
    next_auto = 0
    placeholder_positional = set()
    placeholder_named = set()

    def _add_key(fmtkey):
        """Returns True if there is an error which should early-exit"""
        nonlocal auto, next_auto
        if fmtkey is None:
            return False
        fmtkey, _, _ = fmtkey.partition('.')
        fmtkey, _, _ = fmtkey.partition('[')
        try:
            fmtkey = int(fmtkey)
        except ValueError:
            pass
        else:
            if auto is True:
                self.report(messages.StringDotFormatMixingAutomatic, node)
                return True
            else:
                auto = False
        if fmtkey == '':
            if auto is False:
                self.report(messages.StringDotFormatMixingAutomatic, node)
                return True
            else:
                auto = True
            fmtkey = next_auto
            next_auto += 1
        if isinstance(fmtkey, int):
            placeholder_positional.add(fmtkey)
        else:
            placeholder_named.add(fmtkey)
        return False
    for _, fmtkey, spec, _ in placeholders:
        if _add_key(fmtkey):
            return
        if spec is not None:
            try:
                spec_placeholders = tuple(parse_format_string(spec))
            except ValueError as e:
                self.report(messages.StringDotFormatInvalidFormat, node, e)
                return
            for _, spec_fmtkey, spec_spec, _ in spec_placeholders:
                if spec_spec is not None and '{' in spec_spec:
                    self.report(messages.StringDotFormatInvalidFormat, node, 'Max string recursion exceeded')
                    return
                if _add_key(spec_fmtkey):
                    return
    if any((isinstance(arg, ast.Starred) for arg in node.args)) or any((kwd.arg is None for kwd in node.keywords)):
        return
    substitution_positional = set(range(len(node.args)))
    substitution_named = {kwd.arg for kwd in node.keywords}
    extra_positional = substitution_positional - placeholder_positional
    extra_named = substitution_named - placeholder_named
    missing_arguments = (placeholder_positional | placeholder_named) - (substitution_positional | substitution_named)
    if extra_positional:
        self.report(messages.StringDotFormatExtraPositionalArguments, node, ', '.join(sorted((str(x) for x in extra_positional))))
    if extra_named:
        self.report(messages.StringDotFormatExtraNamedArguments, node, ', '.join(sorted(extra_named)))
    if missing_arguments:
        self.report(messages.StringDotFormatMissingArgument, node, ', '.join(sorted((str(x) for x in missing_arguments))))