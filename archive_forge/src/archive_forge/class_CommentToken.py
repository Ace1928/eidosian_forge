from __future__ import unicode_literals
class CommentToken(Token):
    __slots__ = ('value', 'pre_done')
    id = '<comment>'

    def __init__(self, value, start_mark, end_mark):
        Token.__init__(self, start_mark, end_mark)
        self.value = value

    def reset(self):
        if hasattr(self, 'pre_done'):
            delattr(self, 'pre_done')

    def __repr__(self):
        v = '{!r}'.format(self.value)
        if SHOWLINES:
            try:
                v += ', line: ' + str(self.start_mark.line)
            except:
                pass
        return 'CommentToken({})'.format(v)