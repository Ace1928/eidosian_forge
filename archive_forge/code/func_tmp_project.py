import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
@pytest.fixture(scope='function')
def tmp_project(tmp_path):
    project = tmp_path / 'test_project'
    project.mkdir()
    return project