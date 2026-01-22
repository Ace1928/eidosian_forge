import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
def as_nifti(self):
    if len(self.storage_instances) < 2:
        raise VolumeError('too few slices')
    d = self.storage_instances[0].dicom()
    if self.bits_allocated != 16:
        raise VolumeError('unsupported bits allocated')
    if self.bits_stored != 12:
        raise VolumeError('unsupported bits stored')
    data = numpy.ndarray((len(self.storage_instances), self.rows, self.columns), dtype=numpy.int16)
    for i, si in enumerate(self.storage_instances):
        if i + 1 != si.instance_number:
            raise InstanceStackError(self, i, si)
        logger.info('reading %d/%d' % (i + 1, len(self.storage_instances)))
        d = self.storage_instances[i].dicom()
        data[i, :, :] = d.pixel_array
    d1 = self.storage_instances[0].dicom()
    dn = self.storage_instances[-1].dicom()
    pdi = d1.PixelSpacing[0]
    pdj = d1.PixelSpacing[0]
    pdk = d1.SpacingBetweenSlices
    cosi = d1.ImageOrientationPatient[0:3]
    cosi[0] = -1 * cosi[0]
    cosi[1] = -1 * cosi[1]
    cosj = d1.ImageOrientationPatient[3:6]
    cosj[0] = -1 * cosj[0]
    cosj[1] = -1 * cosj[1]
    pos_1 = numpy.array(d1.ImagePositionPatient)
    pos_1[0] = -1 * pos_1[0]
    pos_1[1] = -1 * pos_1[1]
    pos_n = numpy.array(dn.ImagePositionPatient)
    pos_n[0] = -1 * pos_n[0]
    pos_n[1] = -1 * pos_n[1]
    cosk = pos_n - pos_1
    cosk = cosk / numpy.linalg.norm(cosk)
    m = ((pdi * cosi[0], pdj * cosj[0], pdk * cosk[0], pos_1[0]), (pdi * cosi[1], pdj * cosj[1], pdk * cosk[1], pos_1[1]), (pdi * cosi[2], pdj * cosj[2], pdk * cosk[2], pos_1[2]), (0, 0, 0, 1))
    m = numpy.array(m, dtype=float)
    hdr = Nifti1Header(endianness='<')
    hdr.set_intent(0)
    hdr.set_qform(m, 1)
    hdr.set_xyzt_units(2, 8)
    hdr.set_data_dtype(numpy.int16)
    hdr.set_data_shape((self.columns, self.rows, len(self.storage_instances)))
    s = BytesIO()
    hdr.write_to(s)
    return s.getvalue() + data.tobytes()