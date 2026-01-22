import sys
import os
import struct
import logging
import numpy as np
class DicomSeries(object):
    """DicomSeries
    This class represents a serie of dicom files (SimpleDicomReader
    objects) that belong together. If these are multiple files, they
    represent the slices of a volume (like for CT or MRI).
    """

    def __init__(self, suid, progressIndicator):
        self._entries = []
        self._suid = suid
        self._info = {}
        self._progressIndicator = progressIndicator

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    @property
    def suid(self):
        return self._suid

    @property
    def shape(self):
        """The shape of the data (nz, ny, nx)."""
        return self._info['shape']

    @property
    def sampling(self):
        """The sampling (voxel distances) of the data (dz, dy, dx)."""
        return self._info['sampling']

    @property
    def info(self):
        """A dictionary containing the information as present in the
        first dicomfile of this serie. None if there are no entries."""
        return self._info

    @property
    def description(self):
        """A description of the dicom series. Used fields are
        PatientName, shape of the data, SeriesDescription, and
        ImageComments.
        """
        info = self.info
        if not info:
            return 'DicomSeries containing %i images' % len(self)
        fields = []
        if 'PatientName' in info:
            fields.append('' + info['PatientName'])
        if self.shape:
            tmp = [str(d) for d in self.shape]
            fields.append('x'.join(tmp))
        if 'SeriesDescription' in info:
            fields.append("'" + info['SeriesDescription'] + "'")
        if 'ImageComments' in info:
            fields.append("'" + info['ImageComments'] + "'")
        return ' '.join(fields)

    def __repr__(self):
        adr = hex(id(self)).upper()
        return '<DicomSeries with %i images at %s>' % (len(self), adr)

    def get_numpy_array(self):
        """Get (load) the data that this DicomSeries represents, and return
        it as a numpy array. If this serie contains multiple images, the
        resulting array is 3D, otherwise it's 2D.
        """
        if len(self) == 0:
            raise ValueError('Serie does not contain any files.')
        elif len(self) == 1:
            return self[0].get_numpy_array()
        if self.info is None:
            raise RuntimeError('Cannot return volume if series not finished.')
        slice = self[0].get_numpy_array()
        vol = np.zeros(self.shape, dtype=slice.dtype)
        vol[0] = slice
        self._progressIndicator.start('loading data', '', len(self))
        for z in range(1, len(self)):
            vol[z] = self[z].get_numpy_array()
            self._progressIndicator.set_progress(z + 1)
        self._progressIndicator.finish()
        import gc
        gc.collect()
        return vol

    def _append(self, dcm):
        self._entries.append(dcm)

    def _sort(self):
        self._entries.sort(key=lambda k: k.InstanceNumber)

    def _finish(self):
        """
        Evaluate the series of dicom files. Together they should make up
        a volumetric dataset. This means the files should meet certain
        conditions. Also some additional information has to be calculated,
        such as the distance between the slices. This method sets the
        attributes for "shape", "sampling" and "info".

        This method checks:
          * that there are no missing files
          * that the dimensions of all images match
          * that the pixel spacing of all images match
        """
        L = self._entries
        if len(L) == 0:
            return
        elif len(L) == 1:
            self._info = L[0].info
            return
        ds1 = L[0]
        distance_sum = 0.0
        dimensions = (ds1.Rows, ds1.Columns)
        sampling = ds1.info['sampling'][:2]
        for index in range(len(L)):
            ds2 = L[index]
            pos1 = float(ds1.ImagePositionPatient[2])
            pos2 = float(ds2.ImagePositionPatient[2])
            distance_sum += abs(pos1 - pos2)
            dimensions2 = (ds2.Rows, ds2.Columns)
            sampling2 = ds2.info['sampling'][:2]
            if dimensions != dimensions2:
                raise ValueError('Dimensions of slices does not match.')
            if sampling != sampling2:
                self._progressIndicator.write('Warn: sampling does not match.')
            ds1 = ds2
        distance_mean = distance_sum / (len(L) - 1)
        self._info = L[0].info.copy()
        self._info['shape'] = (len(L),) + ds2.info['shape']
        self._info['sampling'] = (distance_mean,) + ds2.info['sampling']