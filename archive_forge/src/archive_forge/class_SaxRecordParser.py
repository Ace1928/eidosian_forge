import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
class SaxRecordParser(handler.ContentHandler):

    def __init__(self, callback=None):
        self._level = 0
        self._is_active = False
        self._record = None
        self._tag_level = None
        self._tag_payload = None
        self._tag_feeding = None
        self._callback = callback

    def startElement(self, name, attrs):
        self._level += 1
        if self._is_active is False:
            if name == 'record':
                self._is_active = True
                self._tag_level = self._level
                self._record = {}
                if 'date' in attrs:
                    self._record['date'] = attrs['date']
        elif self._level == self._tag_level + 1:
            if name == 'xref':
                if 'type' in attrs and 'data' in attrs:
                    l = self._record.setdefault(attrs['type'], [])
                    l.append(attrs['data'])
            else:
                self._tag_payload = []
                self._tag_feeding = True
        else:
            self._tag_feeding = False

    def endElement(self, name):
        if self._is_active is True:
            if name == 'record' and self._tag_level == self._level:
                self._is_active = False
                self._tag_level = None
                if hasattr(self._callback, '__call__'):
                    self._callback(self._record)
                self._record = None
            elif self._level == self._tag_level + 1:
                if name != 'xref':
                    self._record[name] = ''.join(self._tag_payload)
                    self._tag_payload = None
                    self._tag_feeding = False
        self._level -= 1

    def characters(self, content):
        if self._tag_feeding is True:
            self._tag_payload.append(content)