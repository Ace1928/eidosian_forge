import unittest
from unittest import mock
import pytest
from ..pkg_info import cmp_pkg_version
@mock.patch(_sched('ATTRIBUTE'), [('3.0.0', [('nibabel.nifti1', 'Nifti1Image', 'affine')])])
def test_unremoved_attr():
    with pytest.raises(AssertionError):
        test_attribute_removal()