import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
class SubplotidValidator(BaseValidator):
    """
    "subplotid": {
        "description": "An id string of a subplot type (given by dflt),
                        optionally followed by an integer >1. e.g. if
                        dflt='geo', we can have 'geo', 'geo2', 'geo3',
                        ...",
        "requiredOpts": [
            "dflt"
        ],
        "otherOpts": [
            "regex"
        ]
    }
    """

    def __init__(self, plotly_name, parent_name, dflt=None, regex=None, **kwargs):
        if dflt is None and regex is None:
            raise ValueError('One or both of regex and deflt must be specified')
        super(SubplotidValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        if dflt is not None:
            self.base = dflt
        else:
            self.base = re.match('/\\^(\\w+)', regex).group(1)
        self.regex = self.base + '(\\d*)'

    def description(self):
        desc = "    The '{plotly_name}' property is an identifier of a particular\n    subplot, of type '{base}', that may be specified as the string '{base}'\n    optionally followed by an integer >= 1\n    (e.g. '{base}', '{base}1', '{base}2', '{base}3', etc.)\n        ".format(plotly_name=self.plotly_name, base=self.base)
        return desc

    def validate_coerce(self, v):
        if v is None:
            pass
        elif not isinstance(v, str):
            self.raise_invalid_val(v)
        else:
            match = fullmatch(self.regex, v)
            if not match:
                is_valid = False
            else:
                digit_str = match.group(1)
                if len(digit_str) > 0 and int(digit_str) == 0:
                    is_valid = False
                elif len(digit_str) > 0 and int(digit_str) == 1:
                    v = self.base
                    is_valid = True
                else:
                    is_valid = True
            if not is_valid:
                self.raise_invalid_val(v)
        return v