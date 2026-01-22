from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _python_value_to_header_str(cls, value):
    if isinstance(value, str):
        header_str = value
    else:
        if hasattr(value, 'items'):
            value = sorted(value.items(), key=lambda item: item[1], reverse=True)
        if isinstance(value, (tuple, list)):
            result = []
            for element in value:
                if isinstance(element, (tuple, list)):
                    element = _item_qvalue_pair_to_header_element(pair=element)
                result.append(element)
            header_str = ', '.join(result)
        else:
            header_str = str(value)
    return header_str