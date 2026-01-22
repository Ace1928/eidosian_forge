from __future__ import annotations
from numbers import Integral
import numpy as np
from .externals.netcdf import netcdf_file
from .fileslice import canonical_slicers
from .spatialimages import SpatialHeader, SpatialImage
class Minc1Image(SpatialImage):
    """Class for MINC1 format images

    The MINC1 image class uses the default header type, rather than a specific
    MINC header type - and reads the relevant information from the MINC file on
    load.
    """
    header_class: type[MincHeader] = Minc1Header
    header: MincHeader
    _meta_sniff_len: int = 4
    valid_exts: tuple[str, ...] = ('.mnc',)
    files_types: tuple[tuple[str, str], ...] = (('image', '.mnc'),)
    _compressed_suffixes: tuple[str, ...] = ('.gz', '.bz2', '.zst')
    makeable = True
    rw = False
    ImageArrayProxy = MincImageArrayProxy

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        with file_map['image'].get_prepare_fileobj() as fobj:
            minc_file = Minc1File(netcdf_file(fobj))
            affine = minc_file.get_affine()
            if affine.shape != (4, 4):
                raise MincError('Image does not have 3 spatial dimensions')
            data_dtype = minc_file.get_data_dtype()
            shape = minc_file.get_data_shape()
            zooms = minc_file.get_zooms()
            header = klass.header_class(data_dtype, shape, zooms)
            data = klass.ImageArrayProxy(minc_file)
        return klass(data, affine, header, extra=None, file_map=file_map)