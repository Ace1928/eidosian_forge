import os
import sys
from enum import Enum, _simple_enum
def uuid4():
    """Generate a random UUID."""
    return UUID(bytes=os.urandom(16), version=4)