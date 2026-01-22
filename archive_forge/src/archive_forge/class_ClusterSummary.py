from boto.resultset import ResultSet
class ClusterSummary(EmrObject):
    Fields = set(['Id', 'Name', 'NormalizedInstanceHours'])

    def __init__(self, connection):
        self.connection = connection
        self.status = None

    def startElement(self, name, attrs, connection):
        if name == 'Status':
            self.status = ClusterStatus()
            return self.status
        else:
            return None