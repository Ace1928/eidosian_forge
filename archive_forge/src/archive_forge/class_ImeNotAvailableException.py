from typing import Optional
from typing import Sequence
class ImeNotAvailableException(WebDriverException):
    """Thrown when IME support is not available.

    This exception is thrown for every IME-related method call if IME
    support is not available on the machine.
    """