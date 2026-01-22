import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
class FIBaseBitmap(object):

    def __init__(self, fi, filename, ftype, flags):
        self._fi = fi
        self._filename = filename
        self._ftype = ftype
        self._flags = flags
        self._bitmap = None
        self._close_funcs = []

    def __del__(self):
        self.close()

    def close(self):
        if self._bitmap is not None and self._close_funcs:
            for close_func in self._close_funcs:
                try:
                    with self._fi:
                        fun = close_func[0]
                        fun(*close_func[1:])
                except Exception:
                    pass
            self._close_funcs = []
            self._bitmap = None

    def _set_bitmap(self, bitmap, close_func=None):
        """Function to set the bitmap and specify the function to unload it."""
        if self._bitmap is not None:
            pass
        if close_func is None:
            close_func = (self._fi.lib.FreeImage_Unload, bitmap)
        self._bitmap = bitmap
        if close_func:
            self._close_funcs.append(close_func)

    def get_meta_data(self):
        models = [(name[5:], number) for name, number in METADATA_MODELS.__dict__.items() if name.startswith('FIMD_')]
        metadata = Dict()
        tag = ctypes.c_void_p()
        with self._fi as lib:
            for model_name, number in models:
                mdhandle = lib.FreeImage_FindFirstMetadata(number, self._bitmap, ctypes.byref(tag))
                mdhandle = ctypes.c_void_p(mdhandle)
                if mdhandle:
                    more = True
                    while more:
                        tag_name = lib.FreeImage_GetTagKey(tag).decode('utf-8')
                        tag_type = lib.FreeImage_GetTagType(tag)
                        byte_size = lib.FreeImage_GetTagLength(tag)
                        char_ptr = ctypes.c_char * byte_size
                        data = char_ptr.from_address(lib.FreeImage_GetTagValue(tag))
                        tag_bytes = bytes(bytearray(data))
                        tag_val = tag_bytes
                        if tag_type == METADATA_DATATYPE.FIDT_ASCII:
                            tag_val = tag_bytes.decode('utf-8', 'replace')
                        elif tag_type in METADATA_DATATYPE.dtypes:
                            dtype = METADATA_DATATYPE.dtypes[tag_type]
                            if IS_PYPY and isinstance(dtype, (list, tuple)):
                                pass
                            else:
                                try:
                                    tag_val = numpy.frombuffer(tag_bytes, dtype=dtype).copy()
                                    if len(tag_val) == 1:
                                        tag_val = tag_val[0]
                                except Exception:
                                    pass
                        subdict = metadata.setdefault(model_name, Dict())
                        subdict[tag_name] = tag_val
                        more = lib.FreeImage_FindNextMetadata(mdhandle, ctypes.byref(tag))
                    lib.FreeImage_FindCloseMetadata(mdhandle)
            return metadata

    def set_meta_data(self, metadata):
        models = {}
        for name, number in METADATA_MODELS.__dict__.items():
            if name.startswith('FIMD_'):
                models[name[5:]] = number

        def get_tag_type_number(dtype):
            for number, numpy_dtype in METADATA_DATATYPE.dtypes.items():
                if dtype == numpy_dtype:
                    return number
            else:
                return None
        with self._fi as lib:
            for model_name, subdict in metadata.items():
                number = models.get(model_name, None)
                if number is None:
                    continue
                for tag_name, tag_val in subdict.items():
                    tag = lib.FreeImage_CreateTag()
                    tag = ctypes.c_void_p(tag)
                    try:
                        is_ascii = False
                        if isinstance(tag_val, str):
                            try:
                                tag_bytes = tag_val.encode('ascii')
                                is_ascii = True
                            except UnicodeError:
                                pass
                        if is_ascii:
                            tag_type = METADATA_DATATYPE.FIDT_ASCII
                            tag_count = len(tag_bytes)
                        else:
                            if not hasattr(tag_val, 'dtype'):
                                tag_val = numpy.array([tag_val])
                            tag_type = get_tag_type_number(tag_val.dtype)
                            if tag_type is None:
                                logger.warning('imageio.freeimage warning: Could not determine tag type of %r.' % tag_name)
                                continue
                            tag_bytes = tag_val.tobytes()
                            tag_count = tag_val.size
                        lib.FreeImage_SetTagKey(tag, tag_name.encode('utf-8'))
                        lib.FreeImage_SetTagType(tag, tag_type)
                        lib.FreeImage_SetTagCount(tag, tag_count)
                        lib.FreeImage_SetTagLength(tag, len(tag_bytes))
                        lib.FreeImage_SetTagValue(tag, tag_bytes)
                        tag_key = lib.FreeImage_GetTagKey(tag)
                        lib.FreeImage_SetMetadata(number, self._bitmap, tag_key, tag)
                    except Exception as err:
                        logger.warning('imagio.freeimage warning: Could not set tag %r: %s, %s' % (tag_name, self._fi._get_error_message(), str(err)))
                    finally:
                        lib.FreeImage_DeleteTag(tag)