from antlr3.constants import EOF, DEFAULT_CHANNEL, INVALID_TOKEN_TYPE
class ClassicToken(Token):
    """@brief Alternative token implementation.

    A Token object like we'd use in ANTLR 2.x; has an actual string created
    and associated with this object.  These objects are needed for imaginary
    tree nodes that have payload objects.  We need to create a Token object
    that has a string; the tree node will point at this token.  CommonToken
    has indexes into a char stream and hence cannot be used to introduce
    new strings.
    """

    def __init__(self, type=None, text=None, channel=DEFAULT_CHANNEL, oldToken=None):
        Token.__init__(self)
        if oldToken is not None:
            self.text = oldToken.text
            self.type = oldToken.type
            self.line = oldToken.line
            self.charPositionInLine = oldToken.charPositionInLine
            self.channel = oldToken.channel
        self.text = text
        self.type = type
        self.line = None
        self.charPositionInLine = None
        self.channel = channel
        self.index = None

    def getText(self):
        return self.text

    def setText(self, text):
        self.text = text

    def getType(self):
        return self.type

    def setType(self, ttype):
        self.type = ttype

    def getLine(self):
        return self.line

    def setLine(self, line):
        self.line = line

    def getCharPositionInLine(self):
        return self.charPositionInLine

    def setCharPositionInLine(self, pos):
        self.charPositionInLine = pos

    def getChannel(self):
        return self.channel

    def setChannel(self, channel):
        self.channel = channel

    def getTokenIndex(self):
        return self.index

    def setTokenIndex(self, index):
        self.index = index

    def getInputStream(self):
        return None

    def setInputStream(self, input):
        pass

    def toString(self):
        channelStr = ''
        if self.channel > 0:
            channelStr = ',channel=' + str(self.channel)
        txt = self.text
        if txt is None:
            txt = '<no text>'
        return '[@%r,%r,<%r>%s,%r:%r]' % (self.index, txt, self.type, channelStr, self.line, self.charPositionInLine)
    __str__ = toString
    __repr__ = toString