import enum
import typing
class NativeError(Exception):
    """Stub for a native error that can be used to generate a SpnegoError from a known platform native code."""

    def __init__(self, msg: str, **kwargs: typing.Any) -> None:
        self.msg = msg
        for key in ['maj_code', 'winerror']:
            if key in kwargs:
                setattr(self, key, kwargs[key])