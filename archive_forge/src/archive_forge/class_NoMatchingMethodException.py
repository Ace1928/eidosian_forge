class NoMatchingMethodException(MethodResolutionError):

    def __init__(self, name, receiver):
        super(NoMatchingMethodException, self).__init__(u'No method "{0}" for receiver {1} matches supplied arguments'.format(name, receiver))