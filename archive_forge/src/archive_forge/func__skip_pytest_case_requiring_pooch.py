import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def _skip_pytest_case_requiring_pooch(data_filename):
    """If a test case is calling pooch, skip it.

    This running the test suite in environments without internet
    access, skipping only the tests that try to fetch external data.
    """
    if 'PYTEST_CURRENT_TEST' in os.environ:
        import pytest
        pytest.skip(f'Unable to download {data_filename}', allow_module_level=True)