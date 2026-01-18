import h5py
from h5py._hl.files import make_fapl
import pytest
def test_ros3_s3_fails():
    """ROS3 exceptions for s3:// location"""
    with pytest.raises(ValueError, match='AWS region required for s3:// location'):
        h5py.File('s3://fakebucket/fakekey', 'r', driver='ros3')
    with pytest.raises(ValueError, match='^foo://wrong/scheme: S3 location must begin with'):
        h5py.File('foo://wrong/scheme', 'r', driver='ros3')