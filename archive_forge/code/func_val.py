import hashlib
import os
from enum import Enum, auto
@property
def val(self):
    """Return the output of the lambda against the system's env value."""
    _, default_fn = self.value
    return default_fn(os.getenv(self.name))