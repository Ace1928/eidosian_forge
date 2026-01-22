from datetime import datetime
from boto.compat import six
def _repr_list(self, array):
    result = '['
    for value in array:
        result += ' ' + self._repr_by_type(value) + ','
    if len(result) > 1:
        result = result[:-1] + ' '
    result += ']'
    return result