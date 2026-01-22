class DocumentStartEvent(Event):
    __slots__ = ('explicit', 'version', 'tags')

    def __init__(self, start_mark=None, end_mark=None, explicit=None, version=None, tags=None, comment=None):
        Event.__init__(self, start_mark, end_mark, comment)
        self.explicit = explicit
        self.version = version
        self.tags = tags