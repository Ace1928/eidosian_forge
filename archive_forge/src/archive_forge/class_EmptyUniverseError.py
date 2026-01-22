from typing import Any, Optional
class EmptyUniverseError(ValueError):

    def __init__(self):
        message = 'Universe Domain cannot be an empty string.'
        super().__init__(message)