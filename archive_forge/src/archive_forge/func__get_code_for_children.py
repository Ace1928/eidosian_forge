from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
def _get_code_for_children(self, children, include_prefix):
    if include_prefix:
        return ''.join((c.get_code() for c in children))
    else:
        first = children[0].get_code(include_prefix=False)
        return first + ''.join((c.get_code() for c in children[1:]))