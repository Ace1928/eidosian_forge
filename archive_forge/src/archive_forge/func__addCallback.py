from twisted.positioning import base, ipositioning
def _addCallback(self, name):
    """
        Adds a callback of the given name, setting C{self.called[name]} to
        C{True} when called.
        """

    def callback(*a, **kw):
        self.called[name] = True
    setattr(self, name, callback)