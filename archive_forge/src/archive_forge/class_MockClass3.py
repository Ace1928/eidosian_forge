import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
class MockClass3:

    @deprecated()
    def __init__(self):
        pass