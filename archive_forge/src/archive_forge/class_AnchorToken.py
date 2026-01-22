from __future__ import unicode_literals
class AnchorToken(Token):
    __slots__ = ('value',)
    id = '<anchor>'

    def __init__(self, value, start_mark, end_mark):
        Token.__init__(self, start_mark, end_mark)
        self.value = value