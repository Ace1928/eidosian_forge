import warnings
from typing import Dict, Optional, Sequence
class UnidecodeError(ValueError):

    def __init__(self, message: str, index: Optional[int]=None) -> None:
        """Raised for Unidecode-related errors.

        The index attribute contains the index of the character that caused
        the error.
        """
        super(UnidecodeError, self).__init__(message)
        self.index = index