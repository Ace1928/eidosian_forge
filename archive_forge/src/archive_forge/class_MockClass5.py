import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
class MockClass5(MockClass1):
    """Inherit from deprecated class but does not call super().__init__."""

    def __init__(self, a):
        self.a = a