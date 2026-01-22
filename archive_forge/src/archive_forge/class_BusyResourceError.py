from trio._util import NoPublicConstructor, final
class BusyResourceError(Exception):
    """Raised when a task attempts to use a resource that some other task is
    already using, and this would lead to bugs and nonsense.

    For example, if two tasks try to send data through the same socket at the
    same time, Trio will raise :class:`BusyResourceError` instead of letting
    the data get scrambled.

    """