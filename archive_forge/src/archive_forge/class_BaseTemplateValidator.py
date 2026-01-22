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
class BaseTemplateValidator(CompoundValidator):

    def __init__(self, plotly_name, parent_name, data_class_str, data_docs, **kwargs):
        super(BaseTemplateValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=data_class_str, data_docs=data_docs, **kwargs)

    def description(self):
        compound_description = super(BaseTemplateValidator, self).description()
        compound_description += "\n      - The name of a registered template where current registered templates\n        are stored in the plotly.io.templates configuration object. The names\n        of all registered templates can be retrieved with:\n            >>> import plotly.io as pio\n            >>> list(pio.templates)  # doctest: +ELLIPSIS\n            ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', ...]\n\n      - A string containing multiple registered template names, joined on '+'\n        characters (e.g. 'template1+template2'). In this case the resulting\n        template is computed by merging together the collection of registered\n        templates"
        return compound_description

    def validate_coerce(self, v, skip_invalid=False):
        import plotly.io as pio
        try:
            if v in pio.templates:
                return copy.deepcopy(pio.templates[v])
            elif isinstance(v, str):
                template_names = v.split('+')
                if all([name in pio.templates for name in template_names]):
                    return pio.templates.merge_templates(*template_names)
        except TypeError:
            pass
        if v == {} or (isinstance(v, self.data_class) and v.to_plotly_json() == {}):
            return self.data_class(data_scatter=[{}])
        return super(BaseTemplateValidator, self).validate_coerce(v, skip_invalid=skip_invalid)