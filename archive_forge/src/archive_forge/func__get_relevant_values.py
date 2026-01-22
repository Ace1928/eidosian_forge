import builtins
import inspect
import io
import keyword
import linecache
import os
import re
import sys
import sysconfig
import tokenize
import traceback
def _get_relevant_values(self, source, frame):
    value = None
    pending = None
    is_attribute = False
    is_valid_value = False
    is_assignment = True
    for token in self._syntax_highlighter.tokenize(source):
        type_, string, (_, col), *_ = token
        if pending is not None:
            if type_ != tokenize.OP or string != '=' or is_assignment:
                yield pending
            pending = None
        if type_ == tokenize.NAME and (not keyword.iskeyword(string)):
            if not is_attribute:
                for variables in (frame.f_locals, frame.f_globals):
                    try:
                        value = variables[string]
                    except KeyError:
                        continue
                    else:
                        is_valid_value = True
                        pending = (col, self._format_value(value))
                        break
            elif is_valid_value:
                try:
                    value = inspect.getattr_static(value, string)
                except AttributeError:
                    is_valid_value = False
                else:
                    yield (col, self._format_value(value))
        elif type_ == tokenize.OP and string == '.':
            is_attribute = True
            is_assignment = False
        elif type_ == tokenize.OP and string == ';':
            is_assignment = True
            is_attribute = False
            is_valid_value = False
        else:
            is_attribute = False
            is_valid_value = False
            is_assignment = False
    if pending is not None:
        yield pending