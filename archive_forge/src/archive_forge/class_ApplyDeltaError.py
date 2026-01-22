import binascii
class ApplyDeltaError(Exception):
    """Indicates that applying a delta failed."""

    def __init__(self, *args, **kwargs) -> None:
        Exception.__init__(self, *args, **kwargs)