class YaqlParsingException(YaqlException):

    def __init__(self, value, position, message):
        self.value = value
        self.position = position
        self.message = message
        super(YaqlParsingException, self).__init__(message)