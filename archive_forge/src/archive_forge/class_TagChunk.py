class TagChunk(Chunk):
    __slots__ = ('tag', 'label')

    def __init__(self, tag: str, label: str=None):
        self.tag = tag
        self.label = label

    def __str__(self):
        if self.label is None:
            return self.tag
        else:
            return self.label + ':' + self.tag