from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _write_mini_sect(self, fp_pos, data, padding=b'\x00'):
    """
        Write given sector to file on disk.

        :param fp_pos: int, file position
        :param data: bytes, sector data
        :param padding: single byte, padding character if data < sector size
        """
    if not isinstance(data, bytes):
        raise TypeError('write_mini_sect: data must be a bytes string')
    if not isinstance(padding, bytes) or len(padding) != 1:
        raise TypeError('write_mini_sect: padding must be a bytes string of 1 char')
    try:
        self.fp.seek(fp_pos)
    except Exception:
        log.debug('write_mini_sect(): fp_pos=%d, filesize=%d' % (fp_pos, self._filesize))
        self._raise_defect(DEFECT_FATAL, 'OLE sector index out of range')
    len_data = len(data)
    if len_data < self.mini_sector_size:
        data += padding * (self.mini_sector_size - len_data)
    if self.mini_sector_size < len_data:
        raise ValueError('Data is larger than sector size')
    self.fp.write(data)