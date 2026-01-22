from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
class TypedLeaf(Leaf):
    __slots__ = ('type',)

    def __init__(self, type, value, start_pos, prefix=''):
        super().__init__(value, start_pos, prefix)
        self.type = type