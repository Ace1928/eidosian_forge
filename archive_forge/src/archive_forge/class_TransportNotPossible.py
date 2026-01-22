class TransportNotPossible(TransportError):
    _fmt = 'Transport operation not possible: %(msg)s %(orig_error)s'