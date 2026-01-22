import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _path_get(self, data, path):
    """Return the nested data at the given path.

        For instance:
            data = {'foo': ['bar', 'baz']}
            path = ['foo', 0]
            ==> 'bar'
        """
    d = data
    for step in path:
        d = d[step]
    return d