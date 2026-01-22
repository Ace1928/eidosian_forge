import pickle
import pytest
import numpy as np
class TestUFuncNoLoopError:

    def test_pickling(self):
        """ Test that _UFuncNoLoopError can be pickled """
        assert isinstance(pickle.dumps(_UFuncNoLoopError), bytes)