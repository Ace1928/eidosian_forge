class JailBreak(BzrError):
    _fmt = "An attempt to access a url outside the server jail was made: '%(url)s'."

    def __init__(self, url):
        BzrError.__init__(self, url=url)