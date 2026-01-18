import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
@pytest.fixture(scope='function')
def tmp_project_with_test_file(tmp_project_with_stubs):
    project = tmp_project_with_stubs
    data_stub = project / 'examples' / 'data' / '.data_stubs' / 'test_data.csv'
    data_stub.write_text(TEST_FILE_CONTENT)
    return project