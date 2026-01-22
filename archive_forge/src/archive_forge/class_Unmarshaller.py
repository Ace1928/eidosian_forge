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
class Unmarshaller:
    """Unmarshal an XML-RPC response, based on incoming XML event
    messages (start, data, end).  Call close() to get the resulting
    data structure.

    Note that this reader is fairly tolerant, and gladly accepts bogus
    XML-RPC data without complaining (but not bogus XML).
    """

    def __init__(self, use_datetime=False, use_builtin_types=False):
        self._type = None
        self._stack = []
        self._marks = []
        self._data = []
        self._value = False
        self._methodname = None
        self._encoding = 'utf-8'
        self.append = self._stack.append
        self._use_datetime = use_builtin_types or use_datetime
        self._use_bytes = use_builtin_types

    def close(self):
        if self._type is None or self._marks:
            raise ResponseError()
        if self._type == 'fault':
            raise Fault(**self._stack[0])
        return tuple(self._stack)

    def getmethodname(self):
        return self._methodname

    def xml(self, encoding, standalone):
        self._encoding = encoding

    def start(self, tag, attrs):
        if ':' in tag:
            tag = tag.split(':')[-1]
        if tag == 'array' or tag == 'struct':
            self._marks.append(len(self._stack))
        self._data = []
        if self._value and tag not in self.dispatch:
            raise ResponseError('unknown tag %r' % tag)
        self._value = tag == 'value'

    def data(self, text):
        self._data.append(text)

    def end(self, tag):
        try:
            f = self.dispatch[tag]
        except KeyError:
            if ':' not in tag:
                return
            try:
                f = self.dispatch[tag.split(':')[-1]]
            except KeyError:
                return
        return f(self, ''.join(self._data))

    def end_dispatch(self, tag, data):
        try:
            f = self.dispatch[tag]
        except KeyError:
            if ':' not in tag:
                return
            try:
                f = self.dispatch[tag.split(':')[-1]]
            except KeyError:
                return
        return f(self, data)
    dispatch = {}

    def end_nil(self, data):
        self.append(None)
        self._value = 0
    dispatch['nil'] = end_nil

    def end_boolean(self, data):
        if data == '0':
            self.append(False)
        elif data == '1':
            self.append(True)
        else:
            raise TypeError('bad boolean value')
        self._value = 0
    dispatch['boolean'] = end_boolean

    def end_int(self, data):
        self.append(int(data))
        self._value = 0
    dispatch['i1'] = end_int
    dispatch['i2'] = end_int
    dispatch['i4'] = end_int
    dispatch['i8'] = end_int
    dispatch['int'] = end_int
    dispatch['biginteger'] = end_int

    def end_double(self, data):
        self.append(float(data))
        self._value = 0
    dispatch['double'] = end_double
    dispatch['float'] = end_double

    def end_bigdecimal(self, data):
        self.append(Decimal(data))
        self._value = 0
    dispatch['bigdecimal'] = end_bigdecimal

    def end_string(self, data):
        if self._encoding:
            data = data.decode(self._encoding)
        self.append(data)
        self._value = 0
    dispatch['string'] = end_string
    dispatch['name'] = end_string

    def end_array(self, data):
        mark = self._marks.pop()
        self._stack[mark:] = [self._stack[mark:]]
        self._value = 0
    dispatch['array'] = end_array

    def end_struct(self, data):
        mark = self._marks.pop()
        dict = {}
        items = self._stack[mark:]
        for i in range(0, len(items), 2):
            dict[items[i]] = items[i + 1]
        self._stack[mark:] = [dict]
        self._value = 0
    dispatch['struct'] = end_struct

    def end_base64(self, data):
        value = Binary()
        value.decode(data.encode('ascii'))
        if self._use_bytes:
            value = value.data
        self.append(value)
        self._value = 0
    dispatch['base64'] = end_base64

    def end_dateTime(self, data):
        value = DateTime()
        value.decode(data)
        if self._use_datetime:
            value = _datetime_type(data)
        self.append(value)
    dispatch['dateTime.iso8601'] = end_dateTime

    def end_value(self, data):
        if self._value:
            self.end_string(data)
    dispatch['value'] = end_value

    def end_params(self, data):
        self._type = 'params'
    dispatch['params'] = end_params

    def end_fault(self, data):
        self._type = 'fault'
    dispatch['fault'] = end_fault

    def end_methodName(self, data):
        if self._encoding:
            data = data.decode(self._encoding)
        self._methodname = data
        self._type = 'methodName'
    dispatch['methodName'] = end_methodName