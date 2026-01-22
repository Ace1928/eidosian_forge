from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _escape_and_quote_parameter_value(cls, param_value):
    """
        Escape and quote parameter value where necessary.

        For media type and extension parameter values.
        """
    if param_value == '':
        param_value = '""'
    else:
        param_value = param_value.replace('\\', '\\\\').replace('"', '\\"')
        if not token_compiled_re.match(param_value):
            param_value = '"' + param_value + '"'
    return param_value