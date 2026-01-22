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
class SrcValidator(BaseValidator):

    def __init__(self, plotly_name, parent_name, **kwargs):
        super(SrcValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.chart_studio = get_module('chart_studio')

    def description(self):
        return "    The '{plotly_name}' property must be specified as a string or\n    as a plotly.grid_objs.Column object".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if v is None:
            pass
        elif isinstance(v, str):
            pass
        elif self.chart_studio and isinstance(v, self.chart_studio.grid_objs.Column):
            v = v.id
        else:
            self.raise_invalid_val(v)
        return v