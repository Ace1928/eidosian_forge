class InconsistentDeltaDelta(InconsistentDelta):
    """Used when we get a delta that is not valid."""
    _fmt = 'An inconsistent delta was supplied: %(delta)r\nreason: %(reason)s'

    def __init__(self, delta, reason):
        BzrError.__init__(self)
        self.delta = delta
        self.reason = reason