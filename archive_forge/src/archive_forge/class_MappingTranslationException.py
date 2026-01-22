class MappingTranslationException(YaqlException):

    def __init__(self):
        super(MappingTranslationException, self).__init__(u'Cannot convert mapping to keyword argument')