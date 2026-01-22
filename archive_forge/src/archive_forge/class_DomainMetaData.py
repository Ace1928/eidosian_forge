from __future__ import print_function
from boto.sdb.queryresultset import SelectResultSet
from boto.compat import six
import sys
from xml.sax.handler import ContentHandler
from threading import Thread
class DomainMetaData(object):

    def __init__(self, domain=None):
        self.domain = domain
        self.item_count = None
        self.item_names_size = None
        self.attr_name_count = None
        self.attr_names_size = None
        self.attr_value_count = None
        self.attr_values_size = None

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'ItemCount':
            self.item_count = int(value)
        elif name == 'ItemNamesSizeBytes':
            self.item_names_size = int(value)
        elif name == 'AttributeNameCount':
            self.attr_name_count = int(value)
        elif name == 'AttributeNamesSizeBytes':
            self.attr_names_size = int(value)
        elif name == 'AttributeValueCount':
            self.attr_value_count = int(value)
        elif name == 'AttributeValuesSizeBytes':
            self.attr_values_size = int(value)
        elif name == 'Timestamp':
            self.timestamp = value
        else:
            setattr(self, name, value)