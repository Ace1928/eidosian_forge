import re
from string import Formatter
class ColoredMessage:

    def __init__(self, tokens):
        self.tokens = tokens
        self.stripped = AnsiParser.strip(tokens)

    def colorize(self, ansi_level):
        return AnsiParser.colorize(self.tokens, ansi_level)