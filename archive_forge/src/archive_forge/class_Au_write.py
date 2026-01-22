from collections import namedtuple
import warnings
class Au_write:

    def __init__(self, f):
        if type(f) == type(''):
            import builtins
            f = builtins.open(f, 'wb')
            self._opened = True
        else:
            self._opened = False
        self.initfp(f)

    def __del__(self):
        if self._file:
            self.close()
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def initfp(self, file):
        self._file = file
        self._framerate = 0
        self._nchannels = 0
        self._sampwidth = 0
        self._framesize = 0
        self._nframes = AUDIO_UNKNOWN_SIZE
        self._nframeswritten = 0
        self._datawritten = 0
        self._datalength = 0
        self._info = b''
        self._comptype = 'ULAW'

    def setnchannels(self, nchannels):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nchannels not in (1, 2, 4):
            raise Error('only 1, 2, or 4 channels supported')
        self._nchannels = nchannels

    def getnchannels(self):
        if not self._nchannels:
            raise Error('number of channels not set')
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if sampwidth not in (1, 2, 3, 4):
            raise Error('bad sample width')
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if not self._framerate:
            raise Error('sample width not specified')
        return self._sampwidth

    def setframerate(self, framerate):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._framerate = framerate

    def getframerate(self):
        if not self._framerate:
            raise Error('frame rate not set')
        return self._framerate

    def setnframes(self, nframes):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nframes < 0:
            raise Error('# of frames cannot be negative')
        self._nframes = nframes

    def getnframes(self):
        return self._nframeswritten

    def setcomptype(self, type, name):
        if type in ('NONE', 'ULAW'):
            self._comptype = type
        else:
            raise Error('unknown compression type')

    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        if self._comptype == 'ULAW':
            return 'CCITT G.711 u-law'
        elif self._comptype == 'ALAW':
            return 'CCITT G.711 A-law'
        else:
            return 'not compressed'

    def setparams(self, params):
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        self.setnchannels(nchannels)
        self.setsampwidth(sampwidth)
        self.setframerate(framerate)
        self.setnframes(nframes)
        self.setcomptype(comptype, compname)

    def getparams(self):
        return _sunau_params(self.getnchannels(), self.getsampwidth(), self.getframerate(), self.getnframes(), self.getcomptype(), self.getcompname())

    def tell(self):
        return self._nframeswritten

    def writeframesraw(self, data):
        if not isinstance(data, (bytes, bytearray)):
            data = memoryview(data).cast('B')
        self._ensure_header_written()
        if self._comptype == 'ULAW':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                import audioop
            data = audioop.lin2ulaw(data, self._sampwidth)
        nframes = len(data) // self._framesize
        self._file.write(data)
        self._nframeswritten = self._nframeswritten + nframes
        self._datawritten = self._datawritten + len(data)

    def writeframes(self, data):
        self.writeframesraw(data)
        if self._nframeswritten != self._nframes or self._datalength != self._datawritten:
            self._patchheader()

    def close(self):
        if self._file:
            try:
                self._ensure_header_written()
                if self._nframeswritten != self._nframes or self._datalength != self._datawritten:
                    self._patchheader()
                self._file.flush()
            finally:
                file = self._file
                self._file = None
                if self._opened:
                    file.close()

    def _ensure_header_written(self):
        if not self._nframeswritten:
            if not self._nchannels:
                raise Error('# of channels not specified')
            if not self._sampwidth:
                raise Error('sample width not specified')
            if not self._framerate:
                raise Error('frame rate not specified')
            self._write_header()

    def _write_header(self):
        if self._comptype == 'NONE':
            if self._sampwidth == 1:
                encoding = AUDIO_FILE_ENCODING_LINEAR_8
                self._framesize = 1
            elif self._sampwidth == 2:
                encoding = AUDIO_FILE_ENCODING_LINEAR_16
                self._framesize = 2
            elif self._sampwidth == 3:
                encoding = AUDIO_FILE_ENCODING_LINEAR_24
                self._framesize = 3
            elif self._sampwidth == 4:
                encoding = AUDIO_FILE_ENCODING_LINEAR_32
                self._framesize = 4
            else:
                raise Error('internal error')
        elif self._comptype == 'ULAW':
            encoding = AUDIO_FILE_ENCODING_MULAW_8
            self._framesize = 1
        else:
            raise Error('internal error')
        self._framesize = self._framesize * self._nchannels
        _write_u32(self._file, AUDIO_FILE_MAGIC)
        header_size = 25 + len(self._info)
        header_size = header_size + 7 & ~7
        _write_u32(self._file, header_size)
        if self._nframes == AUDIO_UNKNOWN_SIZE:
            length = AUDIO_UNKNOWN_SIZE
        else:
            length = self._nframes * self._framesize
        try:
            self._form_length_pos = self._file.tell()
        except (AttributeError, OSError):
            self._form_length_pos = None
        _write_u32(self._file, length)
        self._datalength = length
        _write_u32(self._file, encoding)
        _write_u32(self._file, self._framerate)
        _write_u32(self._file, self._nchannels)
        self._file.write(self._info)
        self._file.write(b'\x00' * (header_size - len(self._info) - 24))

    def _patchheader(self):
        if self._form_length_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._form_length_pos)
        _write_u32(self._file, self._datawritten)
        self._datalength = self._datawritten
        self._file.seek(0, 2)