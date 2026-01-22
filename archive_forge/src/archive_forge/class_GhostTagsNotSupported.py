class GhostTagsNotSupported(BzrError):
    _fmt = 'Ghost tags not supported by format %(format)r.'

    def __init__(self, format):
        self.format = format