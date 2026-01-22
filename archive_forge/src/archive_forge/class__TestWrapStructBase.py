import logging
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError
class _TestWrapStructBase:
    """Class implements base tests for binary headers

    It serves as a base class for other binary header tests
    """
    header_class = None

    def get_bad_bb(self):
        return None

    def test_general_init(self):
        hdr = self.header_class()
        binblock = hdr.binaryblock
        assert len(binblock) == hdr.structarr.dtype.itemsize
        assert hdr.endianness == native_code
        hdr = self.header_class(endianness='swapped')
        assert hdr.endianness == swapped_code
        hdr = self.header_class(check=False)

    def _set_something_into_hdr(self, hdr):
        raise NotImplementedError('Not in base type')

    def test__eq__(self):
        hdr1 = self.header_class()
        hdr2 = self.header_class()
        assert hdr1 == hdr2
        self._set_something_into_hdr(hdr1)
        assert hdr1 != hdr2
        self._set_something_into_hdr(hdr2)
        assert hdr1 == hdr2
        hdr3 = hdr2.as_byteswapped()
        assert hdr2 == hdr3
        assert hdr1 != None
        assert hdr1 != 1

    def test_to_from_fileobj(self):
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        str_io.seek(0)
        hdr2 = self.header_class.from_fileobj(str_io)
        assert hdr2.endianness == native_code
        assert hdr2.binaryblock == hdr.binaryblock

    def test_mappingness(self):
        hdr = self.header_class()
        with pytest.raises(ValueError):
            hdr['nonexistent key'] = 0.1
        hdr_dt = hdr.structarr.dtype
        keys = hdr.keys()
        assert keys == list(hdr)
        vals = hdr.values()
        assert len(vals) == len(keys)
        assert keys == list(hdr_dt.names)
        for key, val in hdr.items():
            assert_array_equal(hdr[key], val)
        assert hdr.get('nonexistent key') is None
        assert hdr.get('nonexistent key', 'default') == 'default'
        assert hdr.get(keys[0]) == vals[0]
        assert hdr.get(keys[0], 'default') == vals[0]
        falsyval = 0 if np.issubdtype(hdr_dt[0], np.number) else b''
        hdr[keys[0]] = falsyval
        assert hdr[keys[0]] == falsyval
        assert hdr.get(keys[0]) == falsyval
        assert hdr.get(keys[0], -1) == falsyval

    def test_endianness_ro(self):
        """Its use in initialization tested in the init tests.
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization (or occasionally byteswapping the
        data) - but this is done via via the as_byteswapped method
        """
        hdr = self.header_class()
        with pytest.raises(AttributeError):
            hdr.endianness = '<'

    def test_endian_guess(self):
        eh = self.header_class()
        assert eh.endianness == native_code
        hdr_data = eh.structarr.copy()
        hdr_data = hdr_data.byteswap(swapped_code)
        eh_swapped = self.header_class(hdr_data.tobytes())
        assert eh_swapped.endianness == swapped_code

    def test_binblock_is_file(self):
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        assert str_io.getvalue() == hdr.binaryblock

    def test_structarr(self):
        hdr = self.header_class()
        hdr.structarr
        with pytest.raises(AttributeError):
            hdr.structarr = 0

    def log_chk(self, hdr, level):
        return log_chk(hdr, level)

    def assert_no_log_err(self, hdr):
        """Assert that no logging or errors result from this `hdr`"""
        fhdr, message, raiser = self.log_chk(hdr, 0)
        assert (fhdr, message) == (hdr, '')

    def test_bytes(self):
        hdr1 = self.header_class()
        bb = hdr1.binaryblock
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        self._set_something_into_hdr(hdr1)
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        with pytest.raises(WrapStructError):
            self.header_class(bb[:-1])
        with pytest.raises(WrapStructError):
            self.header_class(bb + b'\x00')
        bb_bad = self.get_bad_bb()
        if bb_bad is None:
            return
        with imageglobals.LoggingOutputSuppressor():
            with pytest.raises(HeaderDataError):
                self.header_class(bb_bad)
        _ = self.header_class(bb_bad, check=False)

    def test_as_byteswapped(self):
        hdr = self.header_class()
        assert hdr.endianness == native_code
        hdr2 = hdr.as_byteswapped(native_code)
        assert not hdr is hdr2
        hdr_bs = hdr.as_byteswapped(swapped_code)
        assert hdr_bs.endianness == swapped_code
        assert hdr.binaryblock != hdr_bs.binaryblock

        class DC(self.header_class):

            def check_fix(self, *args, **kwargs):
                raise Exception
        with pytest.raises(Exception):
            DC(hdr.binaryblock)
        hdr = DC(hdr.binaryblock, check=False)
        hdr2 = hdr.as_byteswapped(native_code)
        hdr_bs = hdr.as_byteswapped(swapped_code)

    def test_empty_check(self):
        hdr = self.header_class()
        hdr.check_fix(error_level=0)

    def _dxer(self, hdr):
        binblock = hdr.binaryblock
        return self.header_class.diagnose_binaryblock(binblock)

    def test_str(self):
        hdr = self.header_class()
        s1 = str(hdr)
        assert len(s1) > 0