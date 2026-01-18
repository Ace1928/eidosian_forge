import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
@pytest.fixture(scope='function')
def tmp_project_with_stubs(tmp_project_with_examples):
    project = tmp_project_with_examples
    data_stubs = project / 'examples' / 'data' / '.data_stubs'
    data_stubs.mkdir()
    return project