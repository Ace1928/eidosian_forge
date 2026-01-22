class StatusInfo(object):
    """
    Describes a status message.
    """

    def __init__(self, status_type=None, normal=None, status=None, message=None):
        self.status_type = status_type
        self.normal = normal
        self.status = status
        self.message = message

    def __repr__(self):
        return 'StatusInfo:%s' % self.message

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'StatusType':
            self.status_type = value
        elif name == 'Normal':
            if value.lower() == 'true':
                self.normal = True
            else:
                self.normal = False
        elif name == 'Status':
            self.status = value
        elif name == 'Message':
            self.message = value
        else:
            setattr(self, name, value)