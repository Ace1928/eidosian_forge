class DocumentEndEvent(Event):
    __slots__ = ('explicit',)

    def __init__(self, start_mark=None, end_mark=None, explicit=None, comment=None):
        Event.__init__(self, start_mark, end_mark, comment)
        self.explicit = explicit