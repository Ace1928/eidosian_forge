import gettext
import os
import re
import textwrap
import warnings
from . import declarative
def unpack_errors(self, encode_variables=False, dict_char='.', list_char='-'):
    """
        Returns the error as a simple data structure -- lists,
        dictionaries, and strings.

        If ``encode_variables`` is true, then this will return a flat
        dictionary, encoded with variable_encode
        """
    if self.error_list:
        assert not encode_variables, 'You can only encode dictionary errors'
        assert not self.error_dict
        return [item.unpack_errors() if item else item for item in self.error_list]
    if self.error_dict:
        result = {}
        for name, item in self.error_dict.items():
            result[name] = item if isinstance(item, str) else item.unpack_errors()
        if encode_variables:
            from . import variabledecode
            result = variabledecode.variable_encode(result, add_repetitions=False, dict_char=dict_char, list_char=list_char)
            for key in list(result.keys()):
                if not result[key]:
                    del result[key]
        return result
    assert not encode_variables, 'You can only encode dictionary errors'
    return self.msg