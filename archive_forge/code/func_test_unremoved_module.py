import unittest
from unittest import mock
import pytest
from ..pkg_info import cmp_pkg_version
@mock.patch(_sched('MODULE'), [('3.0.0', ['nibabel.nifti1'])])
def test_unremoved_module():
    with pytest.raises(AssertionError):
        test_module_removal()