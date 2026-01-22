class SendfileNotAvailableError(RuntimeError):
    """Sendfile syscall is not available.

    Raised if OS does not support sendfile syscall for given socket or
    file type.
    """