from __future__ import annotations
import io
import typing as ty
from copy import deepcopy
from urllib import request
from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener
@classmethod
def _filemap_from_iobase(klass, io_obj: io.IOBase) -> FileMap:
    """For single-file image types, make a file map with the correct key"""
    if len(klass.files_types) > 1:
        raise NotImplementedError('(de)serialization is undefined for multi-file images')
    return klass.make_file_map({klass.files_types[0][0]: io_obj})