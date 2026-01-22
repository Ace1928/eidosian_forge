import numpy as np
import pytest
from pandas.core.arrays import ExtensionArray
class TestExtensionArray:

    def test_errors(self, data, all_arithmetic_operators):
        op_name = all_arithmetic_operators
        with pytest.raises(AttributeError):
            getattr(data, op_name)