import boto
from boto.utils import find_class, Password
from boto.sdb.db.key import Key
from boto.sdb.db.model import Model
from boto.compat import six, encodebytes
from datetime import datetime
from xml.dom.minidom import getDOMImplementation, parse, parseString, Node
class XMLConverter(object):
    """
    Responsible for converting base Python types to format compatible with underlying
    database.  For SimpleDB, that means everything needs to be converted to a string
    when stored in SimpleDB and from a string when retrieved.

    To convert a value, pass it to the encode or decode method.  The encode method
    will take a Python native value and convert to DB format.  The decode method will
    take a DB format value and convert it to Python native format.  To find the appropriate
    method to call, the generic encode/decode methods will look for the type-specific
    method by searching for a method called "encode_<type name>" or "decode_<type name>".
    """

    def __init__(self, manager):
        self.manager = manager
        self.type_map = {bool: (self.encode_bool, self.decode_bool), int: (self.encode_int, self.decode_int), Model: (self.encode_reference, self.decode_reference), Key: (self.encode_reference, self.decode_reference), Password: (self.encode_password, self.decode_password), datetime: (self.encode_datetime, self.decode_datetime)}
        if six.PY2:
            self.type_map[long] = (self.encode_long, self.decode_long)

    def get_text_value(self, parent_node):
        value = ''
        for node in parent_node.childNodes:
            if node.nodeType == node.TEXT_NODE:
                value += node.data
        return value

    def encode(self, item_type, value):
        if item_type in self.type_map:
            encode = self.type_map[item_type][0]
            return encode(value)
        return value

    def decode(self, item_type, value):
        if item_type in self.type_map:
            decode = self.type_map[item_type][1]
            return decode(value)
        else:
            value = self.get_text_value(value)
        return value

    def encode_prop(self, prop, value):
        if isinstance(value, list):
            if hasattr(prop, 'item_type'):
                new_value = []
                for v in value:
                    item_type = getattr(prop, 'item_type')
                    if Model in item_type.mro():
                        item_type = Model
                    new_value.append(self.encode(item_type, v))
                return new_value
            else:
                return value
        else:
            return self.encode(prop.data_type, value)

    def decode_prop(self, prop, value):
        if prop.data_type == list:
            if hasattr(prop, 'item_type'):
                item_type = getattr(prop, 'item_type')
                if Model in item_type.mro():
                    item_type = Model
                values = []
                for item_node in value.getElementsByTagName('item'):
                    value = self.decode(item_type, item_node)
                    values.append(value)
                return values
            else:
                return self.get_text_value(value)
        else:
            return self.decode(prop.data_type, value)

    def encode_int(self, value):
        value = int(value)
        return '%d' % value

    def decode_int(self, value):
        value = self.get_text_value(value)
        if value:
            value = int(value)
        else:
            value = None
        return value

    def encode_long(self, value):
        value = long(value)
        return '%d' % value

    def decode_long(self, value):
        value = self.get_text_value(value)
        return long(value)

    def encode_bool(self, value):
        if value == True:
            return 'true'
        else:
            return 'false'

    def decode_bool(self, value):
        value = self.get_text_value(value)
        if value.lower() == 'true':
            return True
        else:
            return False

    def encode_datetime(self, value):
        return value.strftime(ISO8601)

    def decode_datetime(self, value):
        value = self.get_text_value(value)
        try:
            return datetime.strptime(value, ISO8601)
        except:
            return None

    def encode_reference(self, value):
        if isinstance(value, six.string_types):
            return value
        if value is None:
            return ''
        else:
            val_node = self.manager.doc.createElement('object')
            val_node.setAttribute('id', value.id)
            val_node.setAttribute('class', '%s.%s' % (value.__class__.__module__, value.__class__.__name__))
            return val_node

    def decode_reference(self, value):
        if not value:
            return None
        try:
            value = value.childNodes[0]
            class_name = value.getAttribute('class')
            id = value.getAttribute('id')
            cls = find_class(class_name)
            return cls.get_by_ids(id)
        except:
            return None

    def encode_password(self, value):
        if value and len(value) > 0:
            return str(value)
        else:
            return None

    def decode_password(self, value):
        value = self.get_text_value(value)
        return Password(value)