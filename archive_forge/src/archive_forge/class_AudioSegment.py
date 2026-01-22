from __future__ import division
import array
import os
import subprocess
from tempfile import TemporaryFile, NamedTemporaryFile
import wave
import sys
import struct
from .logging_utils import log_conversion, log_subprocess_output
from .utils import mediainfo_json, fsdecode
import base64
from collections import namedtuple
from io import BytesIO
from .utils import (
from .exceptions import (
from . import effects
class AudioSegment(object):
    """
    AudioSegments are *immutable* objects representing segments of audio
    that can be manipulated using python code.

    AudioSegments are slicable using milliseconds.
    for example:
        a = AudioSegment.from_mp3(mp3file)
        first_second = a[:1000] # get the first second of an mp3
        slice = a[5000:10000] # get a slice from 5 to 10 seconds of an mp3
    """
    converter = get_encoder_name()

    @classproperty
    def ffmpeg(cls):
        return cls.converter

    @ffmpeg.setter
    def ffmpeg(cls, val):
        cls.converter = val
    DEFAULT_CODECS = {'ogg': 'libvorbis'}

    def __init__(self, data=None, *args, **kwargs):
        self.sample_width = kwargs.pop('sample_width', None)
        self.frame_rate = kwargs.pop('frame_rate', None)
        self.channels = kwargs.pop('channels', None)
        audio_params = (self.sample_width, self.frame_rate, self.channels)
        if isinstance(data, array.array):
            try:
                data = data.tobytes()
            except:
                data = data.tostring()
        if any(audio_params) and None in audio_params:
            raise MissingAudioParameter('Either all audio parameters or no parameter must be specified')
        elif self.sample_width is not None:
            if len(data) % (self.sample_width * self.channels) != 0:
                raise ValueError("data length must be a multiple of '(sample_width * channels)'")
            self.frame_width = self.channels * self.sample_width
            self._data = data
        elif kwargs.get('metadata', False):
            self._data = data
            for attr, val in kwargs.pop('metadata').items():
                setattr(self, attr, val)
        else:
            try:
                data = data if isinstance(data, (basestring, bytes)) else data.read()
            except OSError:
                d = b''
                reader = data.read(2 ** 31 - 1)
                while reader:
                    d += reader
                    reader = data.read(2 ** 31 - 1)
                data = d
            wav_data = read_wav_audio(data)
            if not wav_data:
                raise CouldntDecodeError("Couldn't read wav audio from data")
            self.channels = wav_data.channels
            self.sample_width = wav_data.bits_per_sample // 8
            self.frame_rate = wav_data.sample_rate
            self.frame_width = self.channels * self.sample_width
            self._data = wav_data.raw_data
            if self.sample_width == 1:
                self._data = audioop.bias(self._data, 1, -128)
        if self.sample_width == 3:
            byte_buffer = BytesIO()
            pack_fmt = 'BBB' if isinstance(self._data[0], int) else 'ccc'
            i = iter(self._data)
            padding = {False: b'\x00', True: b'\xff'}
            for b0, b1, b2 in izip(i, i, i):
                byte_buffer.write(padding[b2 > b'\x7f'[0]])
                old_bytes = struct.pack(pack_fmt, b0, b1, b2)
                byte_buffer.write(old_bytes)
            self._data = byte_buffer.getvalue()
            self.sample_width = 4
            self.frame_width = self.channels * self.sample_width
        super(AudioSegment, self).__init__(*args, **kwargs)

    @property
    def raw_data(self):
        """
        public access to the raw audio data as a bytestring
        """
        return self._data

    def get_array_of_samples(self, array_type_override=None):
        """
        returns the raw_data as an array of samples
        """
        if array_type_override is None:
            array_type_override = self.array_type
        return array.array(array_type_override, self._data)

    @property
    def array_type(self):
        return get_array_type(self.sample_width * 8)

    def __len__(self):
        """
        returns the length of this audio segment in milliseconds
        """
        return round(1000 * (self.frame_count() / self.frame_rate))

    def __eq__(self, other):
        try:
            return self._data == other._data
        except:
            return False

    def __hash__(self):
        return hash(AudioSegment) ^ hash((self.channels, self.frame_rate, self.sample_width, self._data))

    def __ne__(self, other):
        return not self == other

    def __iter__(self):
        return (self[i] for i in xrange(len(self)))

    def __getitem__(self, millisecond):
        if isinstance(millisecond, slice):
            if millisecond.step:
                return (self[i:i + millisecond.step] for i in xrange(*millisecond.indices(len(self))))
            start = millisecond.start if millisecond.start is not None else 0
            end = millisecond.stop if millisecond.stop is not None else len(self)
            start = min(start, len(self))
            end = min(end, len(self))
        else:
            start = millisecond
            end = millisecond + 1
        start = self._parse_position(start) * self.frame_width
        end = self._parse_position(end) * self.frame_width
        data = self._data[start:end]
        expected_length = end - start
        missing_frames = (expected_length - len(data)) // self.frame_width
        if missing_frames:
            if missing_frames > self.frame_count(ms=2):
                raise TooManyMissingFrames('You should never be filling in    more than 2 ms with silence here, missing frames: %s' % missing_frames)
            silence = audioop.mul(data[:self.frame_width], self.sample_width, 0)
            data += silence * missing_frames
        return self._spawn(data)

    def get_sample_slice(self, start_sample=None, end_sample=None):
        """
        Get a section of the audio segment by sample index.

        NOTE: Negative indices do *not* address samples backword
        from the end of the audio segment like a python list.
        This is intentional.
        """
        max_val = int(self.frame_count())

        def bounded(val, default):
            if val is None:
                return default
            if val < 0:
                return 0
            if val > max_val:
                return max_val
            return val
        start_i = bounded(start_sample, 0) * self.frame_width
        end_i = bounded(end_sample, max_val) * self.frame_width
        data = self._data[start_i:end_i]
        return self._spawn(data)

    def __add__(self, arg):
        if isinstance(arg, AudioSegment):
            return self.append(arg, crossfade=0)
        else:
            return self.apply_gain(arg)

    def __radd__(self, rarg):
        """
        Permit use of sum() builtin with an iterable of AudioSegments
        """
        if rarg == 0:
            return self
        raise TypeError('Gains must be the second addend after the AudioSegment')

    def __sub__(self, arg):
        if isinstance(arg, AudioSegment):
            raise TypeError("AudioSegment objects can't be subtracted from each other")
        else:
            return self.apply_gain(-arg)

    def __mul__(self, arg):
        """
        If the argument is an AudioSegment, overlay the multiplied audio
        segment.

        If it's a number, just use the string multiply operation to repeat the
        audio.

        The following would return an AudioSegment that contains the
        audio of audio_seg eight times

        `audio_seg * 8`
        """
        if isinstance(arg, AudioSegment):
            return self.overlay(arg, position=0, loop=True)
        else:
            return self._spawn(data=self._data * arg)

    def _spawn(self, data, overrides={}):
        """
        Creates a new audio segment using the metadata from the current one
        and the data passed in. Should be used whenever an AudioSegment is
        being returned by an operation that would alters the current one,
        since AudioSegment objects are immutable.
        """
        if isinstance(data, list):
            data = b''.join(data)
        if isinstance(data, array.array):
            try:
                data = data.tobytes()
            except:
                data = data.tostring()
        if hasattr(data, 'read'):
            if hasattr(data, 'seek'):
                data.seek(0)
            data = data.read()
        metadata = {'sample_width': self.sample_width, 'frame_rate': self.frame_rate, 'frame_width': self.frame_width, 'channels': self.channels}
        metadata.update(overrides)
        return self.__class__(data=data, metadata=metadata)

    @classmethod
    def _sync(cls, *segs):
        channels = max((seg.channels for seg in segs))
        frame_rate = max((seg.frame_rate for seg in segs))
        sample_width = max((seg.sample_width for seg in segs))
        return tuple((seg.set_channels(channels).set_frame_rate(frame_rate).set_sample_width(sample_width) for seg in segs))

    def _parse_position(self, val):
        if val < 0:
            val = len(self) - abs(val)
        val = self.frame_count(ms=len(self)) if val == float('inf') else self.frame_count(ms=val)
        return int(val)

    @classmethod
    def empty(cls):
        return cls(b'', metadata={'channels': 1, 'sample_width': 1, 'frame_rate': 1, 'frame_width': 1})

    @classmethod
    def silent(cls, duration=1000, frame_rate=11025):
        """
        Generate a silent audio segment.
        duration specified in milliseconds (default duration: 1000ms, default frame_rate: 11025).
        """
        frames = int(frame_rate * (duration / 1000.0))
        data = b'\x00\x00' * frames
        return cls(data, metadata={'channels': 1, 'sample_width': 2, 'frame_rate': frame_rate, 'frame_width': 2})

    @classmethod
    def from_mono_audiosegments(cls, *mono_segments):
        if not len(mono_segments):
            raise ValueError('At least one AudioSegment instance is required')
        segs = cls._sync(*mono_segments)
        if segs[0].channels != 1:
            raise ValueError('AudioSegment.from_mono_audiosegments requires all arguments are mono AudioSegment instances')
        channels = len(segs)
        sample_width = segs[0].sample_width
        frame_rate = segs[0].frame_rate
        frame_count = max((int(seg.frame_count()) for seg in segs))
        data = array.array(segs[0].array_type, b'\x00' * (frame_count * sample_width * channels))
        for i, seg in enumerate(segs):
            data[i::channels] = seg.get_array_of_samples()
        return cls(data, channels=channels, sample_width=sample_width, frame_rate=frame_rate)

    @classmethod
    def from_file_using_temporary_files(cls, file, format=None, codec=None, parameters=None, start_second=None, duration=None, **kwargs):
        orig_file = file
        file, close_file = _fd_or_path_or_tempfile(file, 'rb', tempfile=False)
        if format:
            format = format.lower()
            format = AUDIO_FILE_EXT_ALIASES.get(format, format)

        def is_format(f):
            f = f.lower()
            if format == f:
                return True
            if isinstance(orig_file, basestring):
                return orig_file.lower().endswith('.{0}'.format(f))
            if isinstance(orig_file, bytes):
                return orig_file.lower().endswith('.{0}'.format(f).encode('utf8'))
            return False
        if is_format('wav'):
            try:
                obj = cls._from_safe_wav(file)
                if close_file:
                    file.close()
                if start_second is None and duration is None:
                    return obj
                elif start_second is not None and duration is None:
                    return obj[start_second * 1000:]
                elif start_second is None and duration is not None:
                    return obj[:duration * 1000]
                else:
                    return obj[start_second * 1000:(start_second + duration) * 1000]
            except:
                file.seek(0)
        elif is_format('raw') or is_format('pcm'):
            sample_width = kwargs['sample_width']
            frame_rate = kwargs['frame_rate']
            channels = kwargs['channels']
            metadata = {'sample_width': sample_width, 'frame_rate': frame_rate, 'channels': channels, 'frame_width': channels * sample_width}
            obj = cls(data=file.read(), metadata=metadata)
            if close_file:
                file.close()
            if start_second is None and duration is None:
                return obj
            elif start_second is not None and duration is None:
                return obj[start_second * 1000:]
            elif start_second is None and duration is not None:
                return obj[:duration * 1000]
            else:
                return obj[start_second * 1000:(start_second + duration) * 1000]
        input_file = NamedTemporaryFile(mode='wb', delete=False)
        try:
            input_file.write(file.read())
        except OSError:
            input_file.flush()
            input_file.close()
            input_file = NamedTemporaryFile(mode='wb', delete=False, buffering=2 ** 31 - 1)
            if close_file:
                file.close()
            close_file = True
            file = open(orig_file, buffering=2 ** 13 - 1, mode='rb')
            reader = file.read(2 ** 31 - 1)
            while reader:
                input_file.write(reader)
                reader = file.read(2 ** 31 - 1)
        input_file.flush()
        if close_file:
            file.close()
        output = NamedTemporaryFile(mode='rb', delete=False)
        conversion_command = [cls.converter, '-y']
        if format:
            conversion_command += ['-f', format]
        if codec:
            conversion_command += ['-acodec', codec]
        conversion_command += ['-i', input_file.name, '-vn', '-f', 'wav']
        if start_second is not None:
            conversion_command += ['-ss', str(start_second)]
        if duration is not None:
            conversion_command += ['-t', str(duration)]
        conversion_command += [output.name]
        if parameters is not None:
            conversion_command.extend(parameters)
        log_conversion(conversion_command)
        with open(os.devnull, 'rb') as devnull:
            p = subprocess.Popen(conversion_command, stdin=devnull, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_out, p_err = p.communicate()
        log_subprocess_output(p_out)
        log_subprocess_output(p_err)
        try:
            if p.returncode != 0:
                raise CouldntDecodeError('Decoding failed. ffmpeg returned error code: {0}\n\nOutput from ffmpeg/avlib:\n\n{1}'.format(p.returncode, p_err.decode(errors='ignore')))
            obj = cls._from_safe_wav(output)
        finally:
            input_file.close()
            output.close()
            os.unlink(input_file.name)
            os.unlink(output.name)
        if start_second is None and duration is None:
            return obj
        elif start_second is not None and duration is None:
            return obj[0:]
        elif start_second is None and duration is not None:
            return obj[:duration * 1000]
        else:
            return obj[0:duration * 1000]

    @classmethod
    def from_file(cls, file, format=None, codec=None, parameters=None, start_second=None, duration=None, **kwargs):
        orig_file = file
        try:
            filename = fsdecode(file)
        except TypeError:
            filename = None
        file, close_file = _fd_or_path_or_tempfile(file, 'rb', tempfile=False)
        if format:
            format = format.lower()
            format = AUDIO_FILE_EXT_ALIASES.get(format, format)

        def is_format(f):
            f = f.lower()
            if format == f:
                return True
            if filename:
                return filename.lower().endswith('.{0}'.format(f))
            return False
        if is_format('wav'):
            try:
                if start_second is None and duration is None:
                    return cls._from_safe_wav(file)
                elif start_second is not None and duration is None:
                    return cls._from_safe_wav(file)[start_second * 1000:]
                elif start_second is None and duration is not None:
                    return cls._from_safe_wav(file)[:duration * 1000]
                else:
                    return cls._from_safe_wav(file)[start_second * 1000:(start_second + duration) * 1000]
            except:
                file.seek(0)
        elif is_format('raw') or is_format('pcm'):
            sample_width = kwargs['sample_width']
            frame_rate = kwargs['frame_rate']
            channels = kwargs['channels']
            metadata = {'sample_width': sample_width, 'frame_rate': frame_rate, 'channels': channels, 'frame_width': channels * sample_width}
            if start_second is None and duration is None:
                return cls(data=file.read(), metadata=metadata)
            elif start_second is not None and duration is None:
                return cls(data=file.read(), metadata=metadata)[start_second * 1000:]
            elif start_second is None and duration is not None:
                return cls(data=file.read(), metadata=metadata)[:duration * 1000]
            else:
                return cls(data=file.read(), metadata=metadata)[start_second * 1000:(start_second + duration) * 1000]
        conversion_command = [cls.converter, '-y']
        if format:
            conversion_command += ['-f', format]
        if codec:
            conversion_command += ['-acodec', codec]
        read_ahead_limit = kwargs.get('read_ahead_limit', -1)
        if filename:
            conversion_command += ['-i', filename]
            stdin_parameter = None
            stdin_data = None
        else:
            if cls.converter == 'ffmpeg':
                conversion_command += ['-read_ahead_limit', str(read_ahead_limit), '-i', 'cache:pipe:0']
            else:
                conversion_command += ['-i', '-']
            stdin_parameter = subprocess.PIPE
            stdin_data = file.read()
        if codec:
            info = None
        else:
            info = mediainfo_json(orig_file, read_ahead_limit=read_ahead_limit)
        if info:
            audio_streams = [x for x in info['streams'] if x['codec_type'] == 'audio']
            audio_codec = audio_streams[0].get('codec_name')
            if audio_streams[0].get('sample_fmt') == 'fltp' and audio_codec in ['mp3', 'mp4', 'aac', 'webm', 'ogg']:
                bits_per_sample = 16
            else:
                bits_per_sample = audio_streams[0]['bits_per_sample']
            if bits_per_sample == 8:
                acodec = 'pcm_u8'
            else:
                acodec = 'pcm_s%dle' % bits_per_sample
            conversion_command += ['-acodec', acodec]
        conversion_command += ['-vn', '-f', 'wav']
        if start_second is not None:
            conversion_command += ['-ss', str(start_second)]
        if duration is not None:
            conversion_command += ['-t', str(duration)]
        conversion_command += ['-']
        if parameters is not None:
            conversion_command.extend(parameters)
        log_conversion(conversion_command)
        p = subprocess.Popen(conversion_command, stdin=stdin_parameter, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_out, p_err = p.communicate(input=stdin_data)
        if p.returncode != 0 or len(p_out) == 0:
            if close_file:
                file.close()
            raise CouldntDecodeError('Decoding failed. ffmpeg returned error code: {0}\n\nOutput from ffmpeg/avlib:\n\n{1}'.format(p.returncode, p_err.decode(errors='ignore')))
        p_out = bytearray(p_out)
        fix_wav_headers(p_out)
        p_out = bytes(p_out)
        obj = cls(p_out)
        if close_file:
            file.close()
        if start_second is None and duration is None:
            return obj
        elif start_second is not None and duration is None:
            return obj[0:]
        elif start_second is None and duration is not None:
            return obj[:duration * 1000]
        else:
            return obj[0:duration * 1000]

    @classmethod
    def from_mp3(cls, file, parameters=None):
        return cls.from_file(file, 'mp3', parameters=parameters)

    @classmethod
    def from_flv(cls, file, parameters=None):
        return cls.from_file(file, 'flv', parameters=parameters)

    @classmethod
    def from_ogg(cls, file, parameters=None):
        return cls.from_file(file, 'ogg', parameters=parameters)

    @classmethod
    def from_wav(cls, file, parameters=None):
        return cls.from_file(file, 'wav', parameters=parameters)

    @classmethod
    def from_raw(cls, file, **kwargs):
        return cls.from_file(file, 'raw', sample_width=kwargs['sample_width'], frame_rate=kwargs['frame_rate'], channels=kwargs['channels'])

    @classmethod
    def _from_safe_wav(cls, file):
        file, close_file = _fd_or_path_or_tempfile(file, 'rb', tempfile=False)
        file.seek(0)
        obj = cls(data=file)
        if close_file:
            file.close()
        return obj

    def export(self, out_f=None, format='mp3', codec=None, bitrate=None, parameters=None, tags=None, id3v2_version='4', cover=None):
        """
        Export an AudioSegment to a file with given options

        out_f (string):
            Path to destination audio file. Also accepts os.PathLike objects on
            python >= 3.6

        format (string)
            Format for destination audio file.
            ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)

        codec (string)
            Codec used to encode the destination file.

        bitrate (string)
            Bitrate used when encoding destination file. (64, 92, 128, 256, 312k...)
            Each codec accepts different bitrate arguments so take a look at the
            ffmpeg documentation for details (bitrate usually shown as -b, -ba or
            -a:b).

        parameters (list of strings)
            Aditional ffmpeg/avconv parameters

        tags (dict)
            Set metadata information to destination files
            usually used as tags. ({title='Song Title', artist='Song Artist'})

        id3v2_version (string)
            Set ID3v2 version for tags. (default: '4')

        cover (file)
            Set cover for audio file from image file. (png or jpg)
        """
        id3v2_allowed_versions = ['3', '4']
        if format == 'raw' and (codec is not None or parameters is not None):
            raise AttributeError('Can not invoke ffmpeg when export format is "raw"; specify an ffmpeg raw format like format="s16le" instead or call export(format="raw") with no codec or parameters')
        out_f, _ = _fd_or_path_or_tempfile(out_f, 'wb+')
        out_f.seek(0)
        if format == 'raw':
            out_f.write(self._data)
            out_f.seek(0)
            return out_f
        easy_wav = format == 'wav' and codec is None and (parameters is None)
        if easy_wav:
            data = out_f
        else:
            data = NamedTemporaryFile(mode='wb', delete=False)
        pcm_for_wav = self._data
        if self.sample_width == 1:
            pcm_for_wav = audioop.bias(self._data, 1, 128)
        wave_data = wave.open(data, 'wb')
        wave_data.setnchannels(self.channels)
        wave_data.setsampwidth(self.sample_width)
        wave_data.setframerate(self.frame_rate)
        wave_data.setnframes(int(self.frame_count()))
        wave_data.writeframesraw(pcm_for_wav)
        wave_data.close()
        if easy_wav:
            out_f.seek(0)
            return out_f
        output = NamedTemporaryFile(mode='w+b', delete=False)
        conversion_command = [self.converter, '-y', '-f', 'wav', '-i', data.name]
        if codec is None:
            codec = self.DEFAULT_CODECS.get(format, None)
        if cover is not None:
            if cover.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')) and format == 'mp3':
                conversion_command.extend(['-i', cover, '-map', '0', '-map', '1', '-c:v', 'mjpeg'])
            else:
                raise AttributeError('Currently cover images are only supported by MP3 files. The allowed image formats are: .tif, .jpg, .bmp, .jpeg and .png.')
        if codec is not None:
            conversion_command.extend(['-acodec', codec])
        if bitrate is not None:
            conversion_command.extend(['-b:a', bitrate])
        if parameters is not None:
            conversion_command.extend(parameters)
        if tags is not None:
            if not isinstance(tags, dict):
                raise InvalidTag('Tags must be a dictionary.')
            else:
                for key, value in tags.items():
                    conversion_command.extend(['-metadata', '{0}={1}'.format(key, value)])
                if format == 'mp3':
                    if id3v2_version not in id3v2_allowed_versions:
                        raise InvalidID3TagVersion('id3v2_version not allowed, allowed versions: %s' % id3v2_allowed_versions)
                    conversion_command.extend(['-id3v2_version', id3v2_version])
        if sys.platform == 'darwin' and codec == 'mp3':
            conversion_command.extend(['-write_xing', '0'])
        conversion_command.extend(['-f', format, output.name])
        log_conversion(conversion_command)
        with open(os.devnull, 'rb') as devnull:
            p = subprocess.Popen(conversion_command, stdin=devnull, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_out, p_err = p.communicate()
        log_subprocess_output(p_out)
        log_subprocess_output(p_err)
        if p.returncode != 0:
            raise CouldntEncodeError('Encoding failed. ffmpeg/avlib returned error code: {0}\n\nCommand:{1}\n\nOutput from ffmpeg/avlib:\n\n{2}'.format(p.returncode, conversion_command, p_err.decode(errors='ignore')))
        output.seek(0)
        out_f.write(output.read())
        data.close()
        output.close()
        os.unlink(data.name)
        os.unlink(output.name)
        out_f.seek(0)
        return out_f

    def get_frame(self, index):
        frame_start = index * self.frame_width
        frame_end = frame_start + self.frame_width
        return self._data[frame_start:frame_end]

    def frame_count(self, ms=None):
        """
        returns the number of frames for the given number of milliseconds, or
            if not specified, the number of frames in the whole AudioSegment
        """
        if ms is not None:
            return ms * (self.frame_rate / 1000.0)
        else:
            return float(len(self._data) // self.frame_width)

    def set_sample_width(self, sample_width):
        if sample_width == self.sample_width:
            return self
        frame_width = self.channels * sample_width
        return self._spawn(audioop.lin2lin(self._data, self.sample_width, sample_width), overrides={'sample_width': sample_width, 'frame_width': frame_width})

    def set_frame_rate(self, frame_rate):
        if frame_rate == self.frame_rate:
            return self
        if self._data:
            converted, _ = audioop.ratecv(self._data, self.sample_width, self.channels, self.frame_rate, frame_rate, None)
        else:
            converted = self._data
        return self._spawn(data=converted, overrides={'frame_rate': frame_rate})

    def set_channels(self, channels):
        if channels == self.channels:
            return self
        if channels == 2 and self.channels == 1:
            fn = audioop.tostereo
            frame_width = self.frame_width * 2
            fac = 1
            converted = fn(self._data, self.sample_width, fac, fac)
        elif channels == 1 and self.channels == 2:
            fn = audioop.tomono
            frame_width = self.frame_width // 2
            fac = 0.5
            converted = fn(self._data, self.sample_width, fac, fac)
        elif channels == 1:
            channels_data = [seg.get_array_of_samples() for seg in self.split_to_mono()]
            frame_count = int(self.frame_count())
            converted = array.array(channels_data[0].typecode, b'\x00' * (frame_count * self.sample_width))
            for raw_channel_data in channels_data:
                for i in range(frame_count):
                    converted[i] += raw_channel_data[i] // self.channels
            frame_width = self.frame_width // self.channels
        elif self.channels == 1:
            dup_channels = [self for iChannel in range(channels)]
            return AudioSegment.from_mono_audiosegments(*dup_channels)
        else:
            raise ValueError('AudioSegment.set_channels only supports mono-to-multi channel and multi-to-mono channel conversion')
        return self._spawn(data=converted, overrides={'channels': channels, 'frame_width': frame_width})

    def split_to_mono(self):
        if self.channels == 1:
            return [self]
        samples = self.get_array_of_samples()
        mono_channels = []
        for i in range(self.channels):
            samples_for_current_channel = samples[i::self.channels]
            try:
                mono_data = samples_for_current_channel.tobytes()
            except AttributeError:
                mono_data = samples_for_current_channel.tostring()
            mono_channels.append(self._spawn(mono_data, overrides={'channels': 1, 'frame_width': self.sample_width}))
        return mono_channels

    @property
    def rms(self):
        return audioop.rms(self._data, self.sample_width)

    @property
    def dBFS(self):
        rms = self.rms
        if not rms:
            return -float('infinity')
        return ratio_to_db(self.rms / self.max_possible_amplitude)

    @property
    def max(self):
        return audioop.max(self._data, self.sample_width)

    @property
    def max_possible_amplitude(self):
        bits = self.sample_width * 8
        max_possible_val = 2 ** bits
        return max_possible_val / 2

    @property
    def max_dBFS(self):
        return ratio_to_db(self.max, self.max_possible_amplitude)

    @property
    def duration_seconds(self):
        return self.frame_rate and self.frame_count() / self.frame_rate or 0.0

    def get_dc_offset(self, channel=1):
        """
        Returns a value between -1.0 and 1.0 representing the DC offset of a
        channel (1 for left, 2 for right).
        """
        if not 1 <= channel <= 2:
            raise ValueError('channel value must be 1 (left) or 2 (right)')
        if self.channels == 1:
            data = self._data
        elif channel == 1:
            data = audioop.tomono(self._data, self.sample_width, 1, 0)
        else:
            data = audioop.tomono(self._data, self.sample_width, 0, 1)
        return float(audioop.avg(data, self.sample_width)) / self.max_possible_amplitude

    def remove_dc_offset(self, channel=None, offset=None):
        """
        Removes DC offset of given channel. Calculates offset if it's not given.
        Offset values must be in range -1.0 to 1.0. If channel is None, removes
        DC offset from all available channels.
        """
        if channel and (not 1 <= channel <= 2):
            raise ValueError('channel value must be None, 1 (left) or 2 (right)')
        if offset and (not -1.0 <= offset <= 1.0):
            raise ValueError('offset value must be in range -1.0 to 1.0')
        if offset:
            offset = int(round(offset * self.max_possible_amplitude))

        def remove_data_dc(data, off):
            if not off:
                off = audioop.avg(data, self.sample_width)
            return audioop.bias(data, self.sample_width, -off)
        if self.channels == 1:
            return self._spawn(data=remove_data_dc(self._data, offset))
        left_channel = audioop.tomono(self._data, self.sample_width, 1, 0)
        right_channel = audioop.tomono(self._data, self.sample_width, 0, 1)
        if not channel or channel == 1:
            left_channel = remove_data_dc(left_channel, offset)
        if not channel or channel == 2:
            right_channel = remove_data_dc(right_channel, offset)
        left_channel = audioop.tostereo(left_channel, self.sample_width, 1, 0)
        right_channel = audioop.tostereo(right_channel, self.sample_width, 0, 1)
        return self._spawn(data=audioop.add(left_channel, right_channel, self.sample_width))

    def apply_gain(self, volume_change):
        return self._spawn(data=audioop.mul(self._data, self.sample_width, db_to_float(float(volume_change))))

    def overlay(self, seg, position=0, loop=False, times=None, gain_during_overlay=None):
        """
        Overlay the provided segment on to this segment starting at the
        specificed position and using the specfied looping beahvior.

        seg (AudioSegment):
            The audio segment to overlay on to this one.

        position (optional int):
            The position to start overlaying the provided segment in to this
            one.

        loop (optional bool):
            Loop seg as many times as necessary to match this segment's length.
            Overrides loops param.

        times (optional int):
            Loop seg the specified number of times or until it matches this
            segment's length. 1 means once, 2 means twice, ... 0 would make the
            call a no-op
        gain_during_overlay (optional int):
            Changes this segment's volume by the specified amount during the
            duration of time that seg is overlaid on top of it. When negative,
            this has the effect of 'ducking' the audio under the overlay.
        """
        if loop:
            times = -1
        elif times is None:
            times = 1
        elif times == 0:
            return self._spawn(self._data)
        output = StringIO()
        seg1, seg2 = AudioSegment._sync(self, seg)
        sample_width = seg1.sample_width
        spawn = seg1._spawn
        output.write(seg1[:position]._data)
        seg1 = seg1[position:]._data
        seg2 = seg2._data
        pos = 0
        seg1_len = len(seg1)
        seg2_len = len(seg2)
        while times:
            remaining = max(0, seg1_len - pos)
            if seg2_len >= remaining:
                seg2 = seg2[:remaining]
                seg2_len = remaining
                times = 1
            if gain_during_overlay:
                seg1_overlaid = seg1[pos:pos + seg2_len]
                seg1_adjusted_gain = audioop.mul(seg1_overlaid, self.sample_width, db_to_float(float(gain_during_overlay)))
                output.write(audioop.add(seg1_adjusted_gain, seg2, sample_width))
            else:
                output.write(audioop.add(seg1[pos:pos + seg2_len], seg2, sample_width))
            pos += seg2_len
            times -= 1
        output.write(seg1[pos:])
        return spawn(data=output)

    def append(self, seg, crossfade=100):
        seg1, seg2 = AudioSegment._sync(self, seg)
        if not crossfade:
            return seg1._spawn(seg1._data + seg2._data)
        elif crossfade > len(self):
            raise ValueError('Crossfade is longer than the original AudioSegment ({}ms > {}ms)'.format(crossfade, len(self)))
        elif crossfade > len(seg):
            raise ValueError('Crossfade is longer than the appended AudioSegment ({}ms > {}ms)'.format(crossfade, len(seg)))
        xf = seg1[-crossfade:].fade(to_gain=-120, start=0, end=float('inf'))
        xf *= seg2[:crossfade].fade(from_gain=-120, start=0, end=float('inf'))
        output = TemporaryFile()
        output.write(seg1[:-crossfade]._data)
        output.write(xf._data)
        output.write(seg2[crossfade:]._data)
        output.seek(0)
        obj = seg1._spawn(data=output)
        output.close()
        return obj

    def fade(self, to_gain=0, from_gain=0, start=None, end=None, duration=None):
        """
        Fade the volume of this audio segment.

        to_gain (float):
            resulting volume_change in db

        start (int):
            default = beginning of the segment
            when in this segment to start fading in milliseconds

        end (int):
            default = end of the segment
            when in this segment to start fading in milliseconds

        duration (int):
            default = until the end of the audio segment
            the duration of the fade
        """
        if None not in [duration, end, start]:
            raise TypeError('Only two of the three arguments, "start", "end", and "duration" may be specified')
        if to_gain == 0 and from_gain == 0:
            return self
        start = min(len(self), start) if start is not None else None
        end = min(len(self), end) if end is not None else None
        if start is not None and start < 0:
            start += len(self)
        if end is not None and end < 0:
            end += len(self)
        if duration is not None and duration < 0:
            raise InvalidDuration('duration must be a positive integer')
        if duration:
            if start is not None:
                end = start + duration
            elif end is not None:
                start = end - duration
        else:
            duration = end - start
        from_power = db_to_float(from_gain)
        output = []
        before_fade = self[:start]._data
        if from_gain != 0:
            before_fade = audioop.mul(before_fade, self.sample_width, from_power)
        output.append(before_fade)
        gain_delta = db_to_float(to_gain) - from_power
        if duration > 100:
            scale_step = gain_delta / duration
            for i in range(duration):
                volume_change = from_power + scale_step * i
                chunk = self[start + i]
                chunk = audioop.mul(chunk._data, self.sample_width, volume_change)
                output.append(chunk)
        else:
            start_frame = self.frame_count(ms=start)
            end_frame = self.frame_count(ms=end)
            fade_frames = end_frame - start_frame
            scale_step = gain_delta / fade_frames
            for i in range(int(fade_frames)):
                volume_change = from_power + scale_step * i
                sample = self.get_frame(int(start_frame + i))
                sample = audioop.mul(sample, self.sample_width, volume_change)
                output.append(sample)
        after_fade = self[end:]._data
        if to_gain != 0:
            after_fade = audioop.mul(after_fade, self.sample_width, db_to_float(to_gain))
        output.append(after_fade)
        return self._spawn(data=output)

    def fade_out(self, duration):
        return self.fade(to_gain=-120, duration=duration, end=float('inf'))

    def fade_in(self, duration):
        return self.fade(from_gain=-120, duration=duration, start=0)

    def reverse(self):
        return self._spawn(data=audioop.reverse(self._data, self.sample_width))

    def _repr_html_(self):
        src = '\n                    <audio controls>\n                        <source src="data:audio/mpeg;base64,{base64}" type="audio/mpeg"/>\n                        Your browser does not support the audio element.\n                    </audio>\n                  '
        fh = self.export()
        data = base64.b64encode(fh.read()).decode('ascii')
        return src.format(base64=data)