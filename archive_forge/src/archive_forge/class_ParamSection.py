import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class ParamSection(_KVSection):
    """Parser for numpydoc parameter sections.

    E.g. any section that looks like this:
        arg_name
            arg_description
        arg_2 : type, optional
            descriptions can also span...
            ... multiple lines
    """

    def _parse_item(self, key: str, value: str) -> DocstringParam:
        match = PARAM_KEY_REGEX.match(key)
        arg_name = type_name = is_optional = None
        if match is not None:
            arg_name = match.group('name')
            type_name = match.group('type')
            if type_name is not None:
                optional_match = PARAM_OPTIONAL_REGEX.match(type_name)
                if optional_match is not None:
                    type_name = optional_match.group('type')
                    is_optional = True
                else:
                    is_optional = False
        default = None
        if len(value) > 0:
            default_match = PARAM_DEFAULT_REGEX.search(value)
            if default_match is not None:
                default = default_match.group('value')
        return DocstringParam(args=[self.key, arg_name], description=_clean_str(value), arg_name=arg_name, type_name=type_name, is_optional=is_optional, default=default)