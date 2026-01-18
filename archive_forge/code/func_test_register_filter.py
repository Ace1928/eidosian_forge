from ctypes import (
import pytest
import h5py
from h5py import h5z
from .common import insubprocess
def test_register_filter():
    filter_id = 256

    @H5ZFuncT
    def failing_filter_callback(flags, cd_nelemts, cd_values, nbytes, buf_size, buf):
        return 0
    dummy_filter_class = H5ZClass2T(version=h5z.CLASS_T_VERS, id_=filter_id, encoder_present=1, decoder_present=1, name=b'dummy filter', can_apply=None, set_local=None, filter_=failing_filter_callback)
    h5z.register_filter(addressof(dummy_filter_class))
    try:
        assert h5z.filter_avail(filter_id)
        filter_flags = h5z.get_filter_info(filter_id)
        assert filter_flags == h5z.FILTER_CONFIG_ENCODE_ENABLED | h5z.FILTER_CONFIG_DECODE_ENABLED
    finally:
        h5z.unregister_filter(filter_id)
    assert not h5z.filter_avail(filter_id)