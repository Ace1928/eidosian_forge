import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
def __dump(self, value, write):
    try:
        f = self.dispatch[type(value)]
    except KeyError:
        if not hasattr(value, '__dict__'):
            raise TypeError('cannot marshal %s objects' % type(value))
        for type_ in type(value).__mro__:
            if type_ in self.dispatch.keys():
                raise TypeError('cannot marshal %s objects' % type(value))
        f = self.dispatch['_arbitrary_instance']
    f(self, value, write)