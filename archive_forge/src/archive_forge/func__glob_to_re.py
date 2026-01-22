import re
from typing import Iterable, Union
@staticmethod
def _glob_to_re(pattern: str, separator: str='.'):

    def component_to_re(component):
        if '**' in component:
            if component == '**':
                return '(' + re.escape(separator) + '[^' + separator + ']+)*'
            else:
                raise ValueError('** can only appear as an entire path segment')
        else:
            return re.escape(separator) + ('[^' + separator + ']*').join((re.escape(x) for x in component.split('*')))
    result = ''.join((component_to_re(c) for c in pattern.split(separator)))
    return re.compile(result)