import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
class FIBitmap(FIBaseBitmap):
    """Wrapper for the FI bitmap object."""

    def allocate(self, array):
        assert isinstance(array, numpy.ndarray)
        shape = array.shape
        dtype = array.dtype
        r, c = shape[:2]
        if len(shape) == 2:
            n_channels = 1
        elif len(shape) == 3:
            n_channels = shape[2]
        else:
            n_channels = shape[0]
        try:
            fi_type = FI_TYPES.fi_types[dtype.type, n_channels]
            self._fi_type = fi_type
        except KeyError:
            raise ValueError('Cannot write arrays of given type and shape.')
        with self._fi as lib:
            bpp = 8 * dtype.itemsize * n_channels
            bitmap = lib.FreeImage_AllocateT(fi_type, c, r, bpp, 0, 0, 0)
            bitmap = ctypes.c_void_p(bitmap)
            if not bitmap:
                raise RuntimeError('Could not allocate bitmap for storage: %s' % self._fi._get_error_message())
            self._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))

    def load_from_filename(self, filename=None):
        if filename is None:
            filename = self._filename
        with self._fi as lib:
            bitmap = lib.FreeImage_Load(self._ftype, efn(filename), self._flags)
            bitmap = ctypes.c_void_p(bitmap)
            if not bitmap:
                raise ValueError('Could not load bitmap "%s": %s' % (self._filename, self._fi._get_error_message()))
            self._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))

    def save_to_filename(self, filename=None):
        if filename is None:
            filename = self._filename
        ftype = self._ftype
        bitmap = self._bitmap
        fi_type = self._fi_type
        with self._fi as lib:
            if fi_type == FI_TYPES.FIT_BITMAP:
                can_write = lib.FreeImage_FIFSupportsExportBPP(ftype, lib.FreeImage_GetBPP(bitmap))
            else:
                can_write = lib.FreeImage_FIFSupportsExportType(ftype, fi_type)
            if not can_write:
                raise TypeError('Cannot save image of this format to this file type')
            res = lib.FreeImage_Save(ftype, bitmap, efn(filename), self._flags)
            if res is None:
                raise RuntimeError(f'Could not save file `{self._filename}`: {self._fi._get_error_message()}')

    def get_image_data(self):
        dtype, shape, bpp = self._get_type_and_shape()
        array = self._wrap_bitmap_bits_in_array(shape, dtype, False)
        with self._fi as lib:
            isle = lib.FreeImage_IsLittleEndian()

        def n(arr):
            if arr.ndim == 1:
                return arr[::-1].T
            elif arr.ndim == 2:
                return arr[:, ::-1].T
            elif arr.ndim == 3:
                return arr[:, :, ::-1].T
            elif arr.ndim == 4:
                return arr[:, :, :, ::-1].T
        if len(shape) == 3 and isle and (dtype.type == numpy.uint8):
            b = n(array[0])
            g = n(array[1])
            r = n(array[2])
            if shape[0] == 3:
                return numpy.dstack((r, g, b))
            elif shape[0] == 4:
                a = n(array[3])
                return numpy.dstack((r, g, b, a))
            else:
                raise ValueError('Cannot handle images of shape %s' % shape)
        a = n(array).copy()
        return a

    def set_image_data(self, array):
        assert isinstance(array, numpy.ndarray)
        shape = array.shape
        dtype = array.dtype
        with self._fi as lib:
            isle = lib.FreeImage_IsLittleEndian()
        r, c = shape[:2]
        if len(shape) == 2:
            n_channels = 1
            w_shape = (c, r)
        elif len(shape) == 3:
            n_channels = shape[2]
            w_shape = (n_channels, c, r)
        else:
            n_channels = shape[0]

        def n(arr):
            return arr[::-1].T
        wrapped_array = self._wrap_bitmap_bits_in_array(w_shape, dtype, True)
        if len(shape) == 3 and isle and (dtype.type == numpy.uint8):
            R = array[:, :, 0]
            G = array[:, :, 1]
            B = array[:, :, 2]
            wrapped_array[0] = n(B)
            wrapped_array[1] = n(G)
            wrapped_array[2] = n(R)
            if shape[2] == 4:
                A = array[:, :, 3]
                wrapped_array[3] = n(A)
        else:
            wrapped_array[:] = n(array)
        if self._need_finish:
            self._finish_wrapped_array(wrapped_array)
        if len(shape) == 2 and dtype.type == numpy.uint8:
            with self._fi as lib:
                palette = lib.FreeImage_GetPalette(self._bitmap)
            palette = ctypes.c_void_p(palette)
            if not palette:
                raise RuntimeError('Could not get image palette')
            try:
                palette_data = GREY_PALETTE.ctypes.data
            except Exception:
                palette_data = GREY_PALETTE.__array_interface__['data'][0]
            ctypes.memmove(palette, palette_data, 1024)

    def _wrap_bitmap_bits_in_array(self, shape, dtype, save):
        """Return an ndarray view on the data in a FreeImage bitmap. Only
        valid for as long as the bitmap is loaded (if single page) / locked
        in memory (if multipage). This is used in loading data, but
        also during saving, to prepare a strided numpy array buffer.

        """
        with self._fi as lib:
            pitch = lib.FreeImage_GetPitch(self._bitmap)
            bits = lib.FreeImage_GetBits(self._bitmap)
        height = shape[-1]
        byte_size = height * pitch
        itemsize = dtype.itemsize
        if len(shape) == 3:
            strides = (itemsize, shape[0] * itemsize, pitch)
        else:
            strides = (itemsize, pitch)
        data = (ctypes.c_char * byte_size).from_address(bits)
        try:
            self._need_finish = False
            if TEST_NUMPY_NO_STRIDES:
                raise NotImplementedError()
            return numpy.ndarray(shape, dtype=dtype, buffer=data, strides=strides)
        except NotImplementedError:
            if save:
                self._need_finish = True
                return numpy.zeros(shape, dtype=dtype)
            else:
                bb = bytes(bytearray(data))
                array = numpy.frombuffer(bb, dtype=dtype).copy()
                if len(shape) == 3:
                    array.shape = (shape[2], strides[-1] // shape[0], shape[0])
                    array2 = array[:shape[2], :shape[1], :shape[0]]
                    array = numpy.zeros(shape, dtype=array.dtype)
                    for i in range(shape[0]):
                        array[i] = array2[:, :, i].T
                else:
                    array.shape = (shape[1], strides[-1])
                    array = array[:shape[1], :shape[0]].T
                return array

    def _finish_wrapped_array(self, array):
        """Hardcore way to inject numpy array in bitmap."""
        with self._fi as lib:
            pitch = lib.FreeImage_GetPitch(self._bitmap)
            bits = lib.FreeImage_GetBits(self._bitmap)
            bpp = lib.FreeImage_GetBPP(self._bitmap)
        nchannels = bpp // 8 // array.itemsize
        realwidth = pitch // nchannels
        extra = realwidth - array.shape[-2]
        assert 0 <= extra < 10
        newshape = (array.shape[-1], realwidth, nchannels)
        array2 = numpy.zeros(newshape, array.dtype)
        if nchannels == 1:
            array2[:, :array.shape[-2], 0] = array.T
        else:
            for i in range(nchannels):
                array2[:, :array.shape[-2], i] = array[i, :, :].T
        data_ptr = array2.__array_interface__['data'][0]
        ctypes.memmove(bits, data_ptr, array2.nbytes)
        del array2

    def _get_type_and_shape(self):
        bitmap = self._bitmap
        with self._fi as lib:
            w = lib.FreeImage_GetWidth(bitmap)
            h = lib.FreeImage_GetHeight(bitmap)
            self._fi_type = fi_type = lib.FreeImage_GetImageType(bitmap)
            if not fi_type:
                raise ValueError('Unknown image pixel type')
        bpp = None
        dtype = FI_TYPES.dtypes[fi_type]
        if fi_type == FI_TYPES.FIT_BITMAP:
            with self._fi as lib:
                bpp = lib.FreeImage_GetBPP(bitmap)
                has_pallette = lib.FreeImage_GetColorsUsed(bitmap)
            if has_pallette:
                if has_pallette == 256:
                    palette = lib.FreeImage_GetPalette(bitmap)
                    palette = ctypes.c_void_p(palette)
                    p = (ctypes.c_uint8 * (256 * 4)).from_address(palette.value)
                    p = numpy.frombuffer(p, numpy.uint32).copy()
                    if (GREY_PALETTE == p).all():
                        extra_dims = []
                        return (numpy.dtype(dtype), extra_dims + [w, h], bpp)
                newbitmap = lib.FreeImage_ConvertTo32Bits(bitmap)
                newbitmap = ctypes.c_void_p(newbitmap)
                self._set_bitmap(newbitmap)
                return self._get_type_and_shape()
            elif bpp == 8:
                extra_dims = []
            elif bpp == 24:
                extra_dims = [3]
            elif bpp == 32:
                extra_dims = [4]
            else:
                newbitmap = lib.FreeImage_ConvertTo32Bits(bitmap)
                newbitmap = ctypes.c_void_p(newbitmap)
                self._set_bitmap(newbitmap)
                return self._get_type_and_shape()
        else:
            extra_dims = FI_TYPES.extra_dims[fi_type]
        return (numpy.dtype(dtype), extra_dims + [w, h], bpp)

    def quantize(self, quantizer=0, palettesize=256):
        """Quantize the bitmap to make it 8-bit (paletted). Returns a new
        FIBitmap object.
        Only for 24 bit images.
        """
        with self._fi as lib:
            bitmap = lib.FreeImage_ColorQuantizeEx(self._bitmap, quantizer, palettesize, 0, None)
            bitmap = ctypes.c_void_p(bitmap)
            if not bitmap:
                raise ValueError('Could not quantize bitmap "%s": %s' % (self._filename, self._fi._get_error_message()))
            new = FIBitmap(self._fi, self._filename, self._ftype, self._flags)
            new._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))
            new._fi_type = self._fi_type
            return new