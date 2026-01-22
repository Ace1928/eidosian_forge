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
class DataArrayValidator(BaseValidator):
    """
    "data_array": {
        "description": "An {array} of data. The value MUST be an
                        {array}, or we ignore it.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, **kwargs):
        super(DataArrayValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.array_ok = True

    def description(self):
        return "    The '{plotly_name}' property is an array that may be specified as a tuple,\n    list, numpy array, or pandas Series".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if v is None:
            pass
        elif is_homogeneous_array(v):
            v = copy_to_readonly_numpy_array(v)
        elif is_simple_array(v):
            v = to_scalar_or_list(v)
        else:
            self.raise_invalid_val(v)
        return v