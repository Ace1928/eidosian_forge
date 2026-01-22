from boto.resultset import ResultSet
class JobFlowStepList(EmrObject):

    def __ini__(self, connection=None):
        self.connection = connection
        self.stepids = None

    def startElement(self, name, attrs, connection):
        if name == 'StepIds':
            self.stepids = ResultSet([('member', StepId)])
            return self.stepids
        else:
            return None