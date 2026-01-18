from _weakref import (
from _weakrefset import WeakSet, _IterationGuard
import _collections_abc  # Import after _weakref to avoid circular import.
import sys
import itertools
Whether finalizer should be called at exit