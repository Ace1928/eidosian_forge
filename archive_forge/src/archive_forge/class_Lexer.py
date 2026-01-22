import json
from ..error import GraphQLSyntaxError
class Lexer(object):
    __slots__ = ('source', 'prev_position')

    def __init__(self, source):
        self.source = source
        self.prev_position = 0

    def next_token(self, reset_position=None):
        if reset_position is None:
            reset_position = self.prev_position
        token = read_token(self.source, reset_position)
        self.prev_position = token.end
        return token