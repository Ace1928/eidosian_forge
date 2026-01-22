import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
class MockClass2:

    @deprecated('mockclass2_method')
    def method(self):
        pass

    @deprecated('n_features_ is deprecated')
    @property
    def n_features_(self):
        """Number of input features."""
        return 10