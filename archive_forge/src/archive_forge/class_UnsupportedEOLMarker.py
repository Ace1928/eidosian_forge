class UnsupportedEOLMarker(BadBundle):
    _fmt = 'End of line marker was not \\n in bzr revision-bundle'

    def __init__(self):
        BzrError.__init__(self)