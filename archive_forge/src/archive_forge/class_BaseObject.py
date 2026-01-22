from datetime import datetime
from boto.compat import six
class BaseObject(object):

    def __repr__(self):
        result = self.__class__.__name__ + '{ '
        counter = 0
        for key, value in six.iteritems(self.__dict__):
            counter += 1
            if counter > 1:
                result += ', '
            result += key + ': '
            result += self._repr_by_type(value)
        result += ' }'
        return result

    def _repr_by_type(self, value):
        result = ''
        if isinstance(value, Response):
            result += value.__repr__()
        elif isinstance(value, list):
            result += self._repr_list(value)
        else:
            result += str(value)
        return result

    def _repr_list(self, array):
        result = '['
        for value in array:
            result += ' ' + self._repr_by_type(value) + ','
        if len(result) > 1:
            result = result[:-1] + ' '
        result += ']'
        return result