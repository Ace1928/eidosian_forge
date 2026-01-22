class HostedZone(object):

    def __init__(self, id=None, name=None, owner=None, version=None, caller_reference=None):
        self.id = id
        self.name = name
        self.owner = owner
        self.version = version
        self.caller_reference = caller_reference

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Id':
            self.id = value
        elif name == 'Name':
            self.name = value
        elif name == 'Owner':
            self.owner = value
        elif name == 'Version':
            self.version = value
        elif name == 'CallerReference':
            self.caller_reference = value
        else:
            setattr(self, name, value)