from boto.s3.user import User
class BooleanResult(object):

    def __init__(self, marker_elem=None):
        self.status = True
        self.request_id = None
        self.box_usage = None

    def __repr__(self):
        if self.status:
            return 'True'
        else:
            return 'False'

    def __nonzero__(self):
        return self.status

    def startElement(self, name, attrs, connection):
        return None

    def to_boolean(self, value, true_value='true'):
        if value == true_value:
            return True
        else:
            return False

    def endElement(self, name, value, connection):
        if name == 'return':
            self.status = self.to_boolean(value)
        elif name == 'StatusCode':
            self.status = self.to_boolean(value, 'Success')
        elif name == 'IsValid':
            self.status = self.to_boolean(value, 'True')
        elif name == 'RequestId':
            self.request_id = value
        elif name == 'requestId':
            self.request_id = value
        elif name == 'BoxUsage':
            self.request_id = value
        else:
            setattr(self, name, value)