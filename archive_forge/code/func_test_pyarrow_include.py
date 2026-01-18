import os.path
from os.path import join as pjoin
from pyarrow._pyarrow_cpp_tests import get_cpp_tests
def test_pyarrow_include():
    source = os.path.dirname(os.path.abspath(__file__))
    pyarrow_dir = pjoin(source, '..')
    pyarrow_include = pjoin(pyarrow_dir, 'include')
    pyarrow_cpp_include = pjoin(pyarrow_include, 'arrow', 'python')
    assert os.path.exists(pyarrow_include)
    assert os.path.exists(pyarrow_cpp_include)