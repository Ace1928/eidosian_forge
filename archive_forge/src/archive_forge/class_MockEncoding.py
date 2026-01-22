import locale
import pytest
from pandas._config import detect_console_encoding
class MockEncoding:
    """
    Used to add a side effect when accessing the 'encoding' property. If the
    side effect is a str in nature, the value will be returned. Otherwise, the
    side effect should be an exception that will be raised.
    """

    def __init__(self, encoding) -> None:
        super().__init__()
        self.val = encoding

    @property
    def encoding(self):
        return self.raise_or_return(self.val)

    @staticmethod
    def raise_or_return(val):
        if isinstance(val, str):
            return val
        else:
            raise val