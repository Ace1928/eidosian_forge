from collections import namedtuple
import warnings
class Au_read:

    def __init__(self, f):
        if type(f) == type(''):
            import builtins
            f = builtins.open(f, 'rb')
            self._opened = True
        else:
            self._opened = False
        self.initfp(f)

    def __del__(self):
        if self._file:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def initfp(self, file):
        self._file = file
        self._soundpos = 0
        magic = int(_read_u32(file))
        if magic != AUDIO_FILE_MAGIC:
            raise Error('bad magic number')
        self._hdr_size = int(_read_u32(file))
        if self._hdr_size < 24:
            raise Error('header size too small')
        if self._hdr_size > 100:
            raise Error('header size ridiculously large')
        self._data_size = _read_u32(file)
        if self._data_size != AUDIO_UNKNOWN_SIZE:
            self._data_size = int(self._data_size)
        self._encoding = int(_read_u32(file))
        if self._encoding not in _simple_encodings:
            raise Error('encoding not (yet) supported')
        if self._encoding in (AUDIO_FILE_ENCODING_MULAW_8, AUDIO_FILE_ENCODING_ALAW_8):
            self._sampwidth = 2
            self._framesize = 1
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_8:
            self._framesize = self._sampwidth = 1
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_16:
            self._framesize = self._sampwidth = 2
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_24:
            self._framesize = self._sampwidth = 3
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_32:
            self._framesize = self._sampwidth = 4
        else:
            raise Error('unknown encoding')
        self._framerate = int(_read_u32(file))
        self._nchannels = int(_read_u32(file))
        if not self._nchannels:
            raise Error('bad # of channels')
        self._framesize = self._framesize * self._nchannels
        if self._hdr_size > 24:
            self._info = file.read(self._hdr_size - 24)
            self._info, _, _ = self._info.partition(b'\x00')
        else:
            self._info = b''
        try:
            self._data_pos = file.tell()
        except (AttributeError, OSError):
            self._data_pos = None

    def getfp(self):
        return self._file

    def getnchannels(self):
        return self._nchannels

    def getsampwidth(self):
        return self._sampwidth

    def getframerate(self):
        return self._framerate

    def getnframes(self):
        if self._data_size == AUDIO_UNKNOWN_SIZE:
            return AUDIO_UNKNOWN_SIZE
        if self._encoding in _simple_encodings:
            return self._data_size // self._framesize
        return 0

    def getcomptype(self):
        if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
            return 'ULAW'
        elif self._encoding == AUDIO_FILE_ENCODING_ALAW_8:
            return 'ALAW'
        else:
            return 'NONE'

    def getcompname(self):
        if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
            return 'CCITT G.711 u-law'
        elif self._encoding == AUDIO_FILE_ENCODING_ALAW_8:
            return 'CCITT G.711 A-law'
        else:
            return 'not compressed'

    def getparams(self):
        return _sunau_params(self.getnchannels(), self.getsampwidth(), self.getframerate(), self.getnframes(), self.getcomptype(), self.getcompname())

    def getmarkers(self):
        return None

    def getmark(self, id):
        raise Error('no marks')

    def readframes(self, nframes):
        if self._encoding in _simple_encodings:
            if nframes == AUDIO_UNKNOWN_SIZE:
                data = self._file.read()
            else:
                data = self._file.read(nframes * self._framesize)
            self._soundpos += len(data) // self._framesize
            if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=DeprecationWarning)
                    import audioop
                data = audioop.ulaw2lin(data, self._sampwidth)
            return data
        return None

    def rewind(self):
        if self._data_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._data_pos)
        self._soundpos = 0

    def tell(self):
        return self._soundpos

    def setpos(self, pos):
        if pos < 0 or pos > self.getnframes():
            raise Error('position not in range')
        if self._data_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._data_pos + pos * self._framesize)
        self._soundpos = pos

    def close(self):
        file = self._file
        if file:
            self._file = None
            if self._opened:
                file.close()