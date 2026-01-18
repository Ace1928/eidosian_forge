import h5py
from h5py._hl.files import make_fapl
import pytest
@pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 14, 2), reason='AWS S3 access token support in HDF5 >= 1.14.2')
def test_ros3_temp_token():
    """Set and get S3 access token"""
    token = b'#0123FakeToken4567/8/9'
    fapl = make_fapl('ros3', libver=None, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, locking=None, page_buf_size=None, min_meta_keep=None, min_raw_keep=None, alignment_threshold=1, alignment_interval=1, meta_block_size=None, session_token=token)
    assert token, fapl.get_fapl_ros3_token()