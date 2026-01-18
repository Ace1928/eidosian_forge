import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
@pytest.fixture(autouse=True)
def tmp_module(tmp_path):
    """This sets up a temporary directory structure meant to mimic a module
    """
    project = tmp_path / 'static_module'
    project.mkdir()
    examples = project / 'examples'
    examples.mkdir()
    (examples / 'Test_Example_Notebook.ipynb').write_text(EXAMPLE_CONTENT)
    (examples / 'datasets.yml').write_text(DATASETS_CONTENT)
    (examples / 'data').mkdir()
    (examples / 'data' / '.data_stubs').mkdir()
    (examples / 'data' / '.data_stubs' / 'test_data.csv').write_text(TEST_FILE_CONTENT)
    return project