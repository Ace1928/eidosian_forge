class NoParameterFoundException(YaqlException):

    def __init__(self, function_name, param_name):
        message = u"Function '{0}' has no parameter called '{1}'".format(function_name, param_name)
        super(NoParameterFoundException, self).__init__(message)