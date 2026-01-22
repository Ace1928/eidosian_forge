class YaqlLexicalException(YaqlParsingException):

    def __init__(self, value, position):
        msg = u"Lexical error: illegal character '{0}' at position {1}".format(value, position)
        super(YaqlLexicalException, self).__init__(value, position, msg)