class CollectionStartEvent(NodeEvent):
    __slots__ = ('tag', 'implicit', 'flow_style', 'nr_items')

    def __init__(self, anchor, tag, implicit, start_mark=None, end_mark=None, flow_style=None, comment=None, nr_items=None):
        NodeEvent.__init__(self, anchor, start_mark, end_mark, comment)
        self.tag = tag
        self.implicit = implicit
        self.flow_style = flow_style
        self.nr_items = nr_items