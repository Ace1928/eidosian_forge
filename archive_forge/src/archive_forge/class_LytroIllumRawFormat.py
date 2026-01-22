import os
import json
import struct
import logging
import numpy as np
from ..core import Format
from ..v2 import imread
class LytroIllumRawFormat(LytroFormat):
    """This is the Lytro Illum RAW format.
    The raw format is a 10bit image format as used by the Lytro Illum
    light field camera. The format will read the specified raw file and will
    try to load a .txt or .json file with the associated meta data.
    This format does not support writing.


    Parameters for reading
    ----------------------
    meta_only : bool
        Whether to only read the metadata.
    """

    def _can_read(self, request):
        if request.extension in ('.raw',):
            return True

    @staticmethod
    def rearrange_bits(array):
        t0 = array[0::5]
        t1 = array[1::5]
        t2 = array[2::5]
        t3 = array[3::5]
        lsb = array[4::5]
        t0 = np.left_shift(t0, 2) + np.bitwise_and(lsb, 3)
        t1 = np.left_shift(t1, 2) + np.right_shift(np.bitwise_and(lsb, 12), 2)
        t2 = np.left_shift(t2, 2) + np.right_shift(np.bitwise_and(lsb, 48), 4)
        t3 = np.left_shift(t3, 2) + np.right_shift(np.bitwise_and(lsb, 192), 6)
        image = np.zeros(LYTRO_ILLUM_IMAGE_SIZE, dtype=np.uint16)
        image[:, 0::4] = t0.reshape((LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4))
        image[:, 1::4] = t1.reshape((LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4))
        image[:, 2::4] = t2.reshape((LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4))
        image[:, 3::4] = t3.reshape((LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4))
        return np.divide(image, 1023.0).astype(np.float64)

    class Reader(Format.Reader):

        def _open(self, meta_only=False):
            self._file = self.request.get_file()
            self._data = None
            self._meta_only = meta_only

        def _close(self):
            del self._data

        def _get_length(self):
            return 1

        def _get_data(self, index):
            if index not in [0, 'None']:
                raise IndexError('Lytro file contains only one dataset')
            if not self._meta_only:
                if self._data is None:
                    self._data = self._file.read()
                raw = np.frombuffer(self._data, dtype=np.uint8).astype(np.uint16)
                img = LytroIllumRawFormat.rearrange_bits(raw)
            else:
                img = np.array([])
            return (img, self._get_meta_data(index=0))

        def _get_meta_data(self, index):
            if index not in [0, None]:
                raise IndexError('Lytro meta data file contains only one dataset')
            filename_base = os.path.splitext(self.request.get_local_filename())[0]
            meta_data = None
            for ext in ['.txt', '.TXT', '.json', '.JSON']:
                if os.path.isfile(filename_base + ext):
                    meta_data = json.load(open(filename_base + ext))
            if meta_data is not None:
                return meta_data
            else:
                logger.warning('No metadata file found for provided raw file.')
                return {}