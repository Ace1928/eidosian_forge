class NoFunctionRegisteredException(FunctionResolutionError):

    def __init__(self, name):
        super(NoFunctionRegisteredException, self).__init__(u'Unknown function "{0}"'.format(name))