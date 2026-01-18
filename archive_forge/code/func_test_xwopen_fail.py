import pytest
from ase.utils import xwopen
def test_xwopen_fail(tmp_path):
    with pytest.raises(OSError):
        with xwopen(tmp_path / 'does_not_exist/file'):
            pass