class InvalidMethodException(YaqlException):

    def __init__(self, function_name):
        message = u"Function '{0}' cannot be called as a method".format(function_name)
        super(InvalidMethodException, self).__init__(message)