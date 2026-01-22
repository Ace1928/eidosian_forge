from antlr3.constants import EOF, DEFAULT_CHANNEL, INVALID_TOKEN_TYPE
class CommonToken(Token):
    """@brief Basic token implementation.

    This implementation does not copy the text from the input stream upon
    creation, but keeps start/stop pointers into the stream to avoid
    unnecessary copy operations.

    """

    def __init__(self, type=None, channel=DEFAULT_CHANNEL, text=None, input=None, start=None, stop=None, oldToken=None):
        Token.__init__(self)
        if oldToken is not None:
            self.type = oldToken.type
            self.line = oldToken.line
            self.charPositionInLine = oldToken.charPositionInLine
            self.channel = oldToken.channel
            self.index = oldToken.index
            self._text = oldToken._text
            if isinstance(oldToken, CommonToken):
                self.input = oldToken.input
                self.start = oldToken.start
                self.stop = oldToken.stop
        else:
            self.type = type
            self.input = input
            self.charPositionInLine = -1
            self.line = 0
            self.channel = channel
            self.index = -1
            self._text = text
            self.start = start
            self.stop = stop

    def getText(self):
        if self._text is not None:
            return self._text
        if self.input is None:
            return None
        return self.input.substring(self.start, self.stop)

    def setText(self, text):
        """
        Override the text for this token.  getText() will return this text
        rather than pulling from the buffer.  Note that this does not mean
        that start/stop indexes are not valid.  It means that that input
        was converted to a new string in the token object.
        """
        self._text = text
    text = property(getText, setText)

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
        return self.input

    def setInputStream(self, input):
        self.input = input

    def __str__(self):
        if self.type == EOF:
            return '<EOF>'
        channelStr = ''
        if self.channel > 0:
            channelStr = ',channel=' + str(self.channel)
        txt = self.text
        if txt is not None:
            txt = txt.replace('\n', '\\\\n')
            txt = txt.replace('\r', '\\\\r')
            txt = txt.replace('\t', '\\\\t')
        else:
            txt = '<no text>'
        return '[@%d,%d:%d=%r,<%d>%s,%d:%d]' % (self.index, self.start, self.stop, txt, self.type, channelStr, self.line, self.charPositionInLine)