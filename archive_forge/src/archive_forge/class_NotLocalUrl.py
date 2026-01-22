class NotLocalUrl(BzrError):
    _fmt = '%(url)s is not a local path.'

    def __init__(self, url):
        self.url = url