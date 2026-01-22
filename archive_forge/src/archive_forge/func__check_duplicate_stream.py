from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _check_duplicate_stream(self, first_sect, minifat=False):
    """
        Checks if a stream has not been already referenced elsewhere.
        This method should only be called once for each known stream, and only
        if stream size is not null.

        :param first_sect: int, index of first sector of the stream in FAT
        :param minifat: bool, if True, stream is located in the MiniFAT, else in the FAT
        """
    if minifat:
        log.debug('_check_duplicate_stream: sect=%Xh in MiniFAT' % first_sect)
        used_streams = self._used_streams_minifat
    else:
        log.debug('_check_duplicate_stream: sect=%Xh in FAT' % first_sect)
        if first_sect in (DIFSECT, FATSECT, ENDOFCHAIN, FREESECT):
            return
        used_streams = self._used_streams_fat
    if first_sect in used_streams:
        self._raise_defect(DEFECT_INCORRECT, 'Stream referenced twice')
    else:
        used_streams.append(first_sect)