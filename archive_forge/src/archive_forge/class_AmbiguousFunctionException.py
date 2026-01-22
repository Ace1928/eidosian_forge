class AmbiguousFunctionException(FunctionResolutionError):

    def __init__(self, name):
        super(AmbiguousFunctionException, self).__init__(u'Ambiguous function "{0}"'.format(name))