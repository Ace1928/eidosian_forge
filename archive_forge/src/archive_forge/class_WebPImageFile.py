from __future__ import annotations
from io import BytesIO
from . import Image, ImageFile
class WebPImageFile(ImageFile.ImageFile):
    format = 'WEBP'
    format_description = 'WebP image'
    __loaded = 0
    __logical_frame = 0

    def _open(self):
        if not _webp.HAVE_WEBPANIM:
            data, width, height, self._mode, icc_profile, exif = _webp.WebPDecode(self.fp.read())
            if icc_profile:
                self.info['icc_profile'] = icc_profile
            if exif:
                self.info['exif'] = exif
            self._size = (width, height)
            self.fp = BytesIO(data)
            self.tile = [('raw', (0, 0) + self.size, 0, self.mode)]
            self.n_frames = 1
            self.is_animated = False
            return
        self._decoder = _webp.WebPAnimDecoder(self.fp.read())
        width, height, loop_count, bgcolor, frame_count, mode = self._decoder.get_info()
        self._size = (width, height)
        self.info['loop'] = loop_count
        bg_a, bg_r, bg_g, bg_b = (bgcolor >> 24 & 255, bgcolor >> 16 & 255, bgcolor >> 8 & 255, bgcolor & 255)
        self.info['background'] = (bg_r, bg_g, bg_b, bg_a)
        self.n_frames = frame_count
        self.is_animated = self.n_frames > 1
        self._mode = 'RGB' if mode == 'RGBX' else mode
        self.rawmode = mode
        self.tile = []
        icc_profile = self._decoder.get_chunk('ICCP')
        exif = self._decoder.get_chunk('EXIF')
        xmp = self._decoder.get_chunk('XMP ')
        if icc_profile:
            self.info['icc_profile'] = icc_profile
        if exif:
            self.info['exif'] = exif
        if xmp:
            self.info['xmp'] = xmp
        self._reset(reset=False)

    def _getexif(self):
        if 'exif' not in self.info:
            return None
        return self.getexif()._get_merged_dict()

    def getxmp(self):
        """
        Returns a dictionary containing the XMP tags.
        Requires defusedxml to be installed.

        :returns: XMP tags in a dictionary.
        """
        return self._getxmp(self.info['xmp']) if 'xmp' in self.info else {}

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        self.__logical_frame = frame

    def _reset(self, reset=True):
        if reset:
            self._decoder.reset()
        self.__physical_frame = 0
        self.__loaded = -1
        self.__timestamp = 0

    def _get_next(self):
        ret = self._decoder.get_next()
        self.__physical_frame += 1
        if ret is None:
            self._reset()
            self.seek(0)
            msg = 'failed to decode next frame in WebP file'
            raise EOFError(msg)
        data, timestamp = ret
        duration = timestamp - self.__timestamp
        self.__timestamp = timestamp
        timestamp -= duration
        return (data, timestamp, duration)

    def _seek(self, frame):
        if self.__physical_frame == frame:
            return
        if frame < self.__physical_frame:
            self._reset()
        while self.__physical_frame < frame:
            self._get_next()

    def load(self):
        if _webp.HAVE_WEBPANIM:
            if self.__loaded != self.__logical_frame:
                self._seek(self.__logical_frame)
                data, timestamp, duration = self._get_next()
                self.info['timestamp'] = timestamp
                self.info['duration'] = duration
                self.__loaded = self.__logical_frame
                if self.fp and self._exclusive_fp:
                    self.fp.close()
                self.fp = BytesIO(data)
                self.tile = [('raw', (0, 0) + self.size, 0, self.rawmode)]
        return super().load()

    def load_seek(self, pos):
        pass

    def tell(self):
        if not _webp.HAVE_WEBPANIM:
            return super().tell()
        return self.__logical_frame