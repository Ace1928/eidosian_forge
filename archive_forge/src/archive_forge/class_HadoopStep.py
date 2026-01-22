from boto.resultset import ResultSet
class HadoopStep(EmrObject):
    Fields = set(['Id', 'Name', 'ActionOnFailure'])

    def __init__(self, connection=None):
        self.connection = connection
        self.config = None
        self.status = None

    def startElement(self, name, attrs, connection):
        if name == 'Config':
            self.config = StepConfig()
            return self.config
        elif name == 'Status':
            self.status = ClusterStatus()
            return self.status
        else:
            return None