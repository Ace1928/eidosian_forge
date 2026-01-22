from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_object(object):
    literal = 1
    access = 0
    value = None

    def __init__(self, value):
        self.value = value
        self.type = self.__class__.__name__[3:] + 'type'

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__[3:], repr(self.value))