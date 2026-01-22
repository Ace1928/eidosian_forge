import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
class FIMultipageBitmap(FIBaseBitmap):
    """Wrapper for the multipage FI bitmap object."""

    def load_from_filename(self, filename=None):
        if filename is None:
            filename = self._filename
        create_new = False
        read_only = True
        keep_cache_in_memory = False
        with self._fi as lib:
            multibitmap = lib.FreeImage_OpenMultiBitmap(self._ftype, efn(filename), create_new, read_only, keep_cache_in_memory, self._flags)
            multibitmap = ctypes.c_void_p(multibitmap)
            if not multibitmap:
                err = self._fi._get_error_message()
                raise ValueError('Could not open file "%s" as multi-image: %s' % (self._filename, err))
            self._set_bitmap(multibitmap, (lib.FreeImage_CloseMultiBitmap, multibitmap))

    def save_to_filename(self, filename=None):
        if filename is None:
            filename = self._filename
        create_new = True
        read_only = False
        keep_cache_in_memory = False
        with self._fi as lib:
            multibitmap = lib.FreeImage_OpenMultiBitmap(self._ftype, efn(filename), create_new, read_only, keep_cache_in_memory, 0)
            multibitmap = ctypes.c_void_p(multibitmap)
            if not multibitmap:
                msg = 'Could not open file "%s" for writing multi-image: %s' % (self._filename, self._fi._get_error_message())
                raise ValueError(msg)
            self._set_bitmap(multibitmap, (lib.FreeImage_CloseMultiBitmap, multibitmap))

    def __len__(self):
        with self._fi as lib:
            return lib.FreeImage_GetPageCount(self._bitmap)

    def get_page(self, index):
        """Return the sub-bitmap for the given page index.
        Please close the returned bitmap when done.
        """
        with self._fi as lib:
            bitmap = lib.FreeImage_LockPage(self._bitmap, index)
            bitmap = ctypes.c_void_p(bitmap)
            if not bitmap:
                raise ValueError('Could not open sub-image %i in %r: %s' % (index, self._filename, self._fi._get_error_message()))
            bm = FIBitmap(self._fi, self._filename, self._ftype, self._flags)
            bm._set_bitmap(bitmap, (lib.FreeImage_UnlockPage, self._bitmap, bitmap, False))
            return bm

    def append_bitmap(self, bitmap):
        """Add a sub-bitmap to the multi-page bitmap."""
        with self._fi as lib:
            lib.FreeImage_AppendPage(self._bitmap, bitmap._bitmap)