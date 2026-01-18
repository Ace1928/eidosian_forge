import pytest
from ase.utils import deprecated, devnull, tokenize_version
def test_tokenize_version_equal():
    version = '3.8x.xx'
    assert tokenize_version(version) == tokenize_version(version)