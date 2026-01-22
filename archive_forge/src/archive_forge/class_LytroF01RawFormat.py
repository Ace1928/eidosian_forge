import os
import json
import struct
import logging
import numpy as np
from ..core import Format
from ..v2 import imread
class LytroF01RawFormat(LytroFormat):
    """This is the Lytro RAW format for the original F01 Lytro camera.
    The raw format is a 12bit image format as used by the Lytro F01
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
        t0 = array[0::3]
        t1 = array[1::3]
        t2 = array[2::3]
        a0 = np.left_shift(t0, 4) + np.right_shift(np.bitwise_and(t1, 240), 4)
        a1 = np.left_shift(np.bitwise_and(t1, 15), 8) + t2
        image = np.zeros(LYTRO_F01_IMAGE_SIZE, dtype=np.uint16)
        image[:, 0::2] = a0.reshape((LYTRO_F01_IMAGE_SIZE[0], LYTRO_F01_IMAGE_SIZE[1] // 2))
        image[:, 1::2] = a1.reshape((LYTRO_F01_IMAGE_SIZE[0], LYTRO_F01_IMAGE_SIZE[1] // 2))
        return np.divide(image, 4095.0).astype(np.float64)

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
                img = LytroF01RawFormat.rearrange_bits(raw)
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