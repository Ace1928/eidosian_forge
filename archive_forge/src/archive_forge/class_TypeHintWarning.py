from collections import deque
from typing import Deque
class TypeHintWarning(UserWarning):
    """
    A warning that is emitted when a type hint in string form could not be resolved to
    an actual type.
    """