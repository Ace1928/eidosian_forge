class LogFileObject(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.log_filename = None

    def __repr__(self):
        return 'LogFileObject: %s/%s' % (self.dbinstance_id, self.log_filename)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'LogFileData':
            self.data = value
        elif name == 'AdditionalDataPending':
            self.additional_data_pending = value
        elif name == 'Marker':
            self.marker = value
        else:
            setattr(self, name, value)