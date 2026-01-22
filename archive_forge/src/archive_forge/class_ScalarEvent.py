class ScalarEvent(NodeEvent):
    __slots__ = ('tag', 'implicit', 'value', 'style')

    def __init__(self, anchor, tag, implicit, value, start_mark=None, end_mark=None, style=None, comment=None):
        NodeEvent.__init__(self, anchor, start_mark, end_mark, comment)
        self.tag = tag
        self.implicit = implicit
        self.value = value
        self.style = style