import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
def _fix_contents(filename, contents):
    import re
    contents = re.sub('from bytecode', 'from _pydevd_frame_eval.vendored.bytecode', contents, flags=re.MULTILINE)
    contents = re.sub('import bytecode', 'from _pydevd_frame_eval.vendored import bytecode', contents, flags=re.MULTILINE)
    contents = re.sub('def test_version\\(self\\):', 'def skip_test_version(self):', contents, flags=re.MULTILINE)
    if filename.startswith('test_'):
        if 'pytestmark' not in contents:
            pytest_mark = "\nimport pytest\nfrom tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON\nfrom tests_python.debug_constants import TEST_CYTHON\npytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON, reason='Requires CPython >= 3.6')\n"
            contents = pytest_mark + contents
    return contents