from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _form_extension_params_segment(cls, extension_params):
    """
        Convert iterable of extension parameters to str segment for header.

        `extension_params` is an iterable where each item is either a parameter
        string or a (name, value) tuple.
        """
    extension_params_segment = ''
    for item in extension_params:
        try:
            extension_params_segment += ';' + item
        except TypeError:
            param_name, param_value = item
            param_value = cls._escape_and_quote_parameter_value(param_value=param_value)
            extension_params_segment += ';' + param_name + '=' + param_value
    return extension_params_segment