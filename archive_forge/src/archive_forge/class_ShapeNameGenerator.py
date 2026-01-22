from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
class ShapeNameGenerator:
    """Generate unique shape names for a type.

    This class can be used in conjunction with the DenormalizedStructureBuilder
    to generate unique shape names for a given type.

    """

    def __init__(self):
        self._name_cache = defaultdict(int)

    def new_shape_name(self, type_name):
        """Generate a unique shape name.

        This method will guarantee a unique shape name each time it is
        called with the same type.

        ::

            >>> s = ShapeNameGenerator()
            >>> s.new_shape_name('structure')
            'StructureType1'
            >>> s.new_shape_name('structure')
            'StructureType2'
            >>> s.new_shape_name('list')
            'ListType1'
            >>> s.new_shape_name('list')
            'ListType2'


        :type type_name: string
        :param type_name: The type name (structure, list, map, string, etc.)

        :rtype: string
        :return: A unique shape name for the given type

        """
        self._name_cache[type_name] += 1
        current_index = self._name_cache[type_name]
        return f'{type_name.capitalize()}Type{current_index}'