class NotABundle(BzrError):
    _fmt = 'Not a bzr revision-bundle: %(text)r'

    def __init__(self, text):
        BzrError.__init__(self)
        self.text = text