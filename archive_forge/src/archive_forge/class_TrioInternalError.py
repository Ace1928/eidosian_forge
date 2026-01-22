from trio._util import NoPublicConstructor, final
class TrioInternalError(Exception):
    """Raised by :func:`run` if we encounter a bug in Trio, or (possibly) a
    misuse of one of the low-level :mod:`trio.lowlevel` APIs.

    This should never happen! If you get this error, please file a bug.

    Unfortunately, if you get this error it also means that all bets are off –
    Trio doesn't know what is going on and its normal invariants may be void.
    (For example, we might have "lost track" of a task. Or lost track of all
    tasks.) Again, though, this shouldn't happen.

    """