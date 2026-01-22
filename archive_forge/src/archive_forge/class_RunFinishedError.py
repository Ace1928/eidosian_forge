from trio._util import NoPublicConstructor, final
class RunFinishedError(RuntimeError):
    """Raised by `trio.from_thread.run` and similar functions if the
    corresponding call to :func:`trio.run` has already finished.

    """