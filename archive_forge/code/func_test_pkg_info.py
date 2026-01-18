import pytest
import nibabel as nib
from nibabel.pkg_info import cmp_pkg_version
def test_pkg_info():
    """Smoke test nibabel.get_info()

    Hits:
        - nibabel.get_info
        - nibabel.pkg_info.get_pkg_info
        - nibabel.pkg_info.pkg_commit_hash
    """
    info = nib.get_info()