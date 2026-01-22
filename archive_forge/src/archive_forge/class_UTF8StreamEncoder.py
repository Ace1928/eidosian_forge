import struct
import time
import io
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import file_generator
from cherrypy.lib import is_closable_iterator
from cherrypy.lib import set_vary_header
class UTF8StreamEncoder:

    def __init__(self, iterator):
        self._iterator = iterator

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        res = next(self._iterator)
        if isinstance(res, str):
            res = res.encode('utf-8')
        return res

    def close(self):
        if is_closable_iterator(self._iterator):
            self._iterator.close()

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(self, attr)
        return getattr(self._iterator, attr)