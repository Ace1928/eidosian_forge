class MessageAttributeValue(dict):

    def __init__(self, parent):
        self.parent = parent

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'DataType':
            self['data_type'] = value
        elif name == 'StringValue':
            self['string_value'] = value
        elif name == 'BinaryValue':
            self['binary_value'] = value
        elif name == 'StringListValue':
            self['string_list_value'] = value
        elif name == 'BinaryListValue':
            self['binary_list_value'] = value