import os
import unittest
import certifi
def test_py_typed_exists(self) -> None:
    assert os.path.exists(os.path.join(os.path.dirname(certifi.__file__), 'py.typed'))