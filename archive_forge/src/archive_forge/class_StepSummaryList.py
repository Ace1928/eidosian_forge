from boto.resultset import ResultSet
class StepSummaryList(EmrObject):
    Fields = set(['Marker'])

    def __init__(self, connection=None):
        self.connection = connection
        self.steps = None

    def startElement(self, name, attrs, connection):
        if name == 'Steps':
            self.steps = ResultSet([('member', StepSummary)])
            return self.steps
        else:
            return None