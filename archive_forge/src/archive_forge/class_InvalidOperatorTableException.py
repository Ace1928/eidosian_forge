class InvalidOperatorTableException(YaqlException):

    def __init__(self, op):
        super(InvalidOperatorTableException, self).__init__(u"Invalid records in operator table for operator '{0}".format(op))