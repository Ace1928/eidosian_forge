import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _path_set(self, data, path, value):
    """Set the value of a key in the given data.

        Example:
            data = {'foo': ['bar', 'baz']}
            path = ['foo', 1]
            value = 'bin'
            ==> data = {'foo': ['bar', 'bin']}
        """
    container = self._path_get(data, path[:-1])
    container[path[-1]] = value