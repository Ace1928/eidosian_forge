from decimal import Decimal
from boto.compat import filter, map
class ResponseElement(object):

    def __init__(self, connection=None, name=None):
        if connection is not None:
            self._connection = connection
        self._name = name or self.__class__.__name__

    @property
    def connection(self):
        return self._connection

    def __repr__(self):
        render = lambda pair: '{!s}: {!r}'.format(*pair)
        do_show = lambda pair: not pair[0].startswith('_')
        attrs = filter(do_show, self.__dict__.items())
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(map(render, attrs)))

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name != self._name:
            setattr(self, name, value)