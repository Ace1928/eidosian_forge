import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
class SynthesisSource(Source):
    """Base class for synthesized waveforms.

    :Parameters:
        `generator` : A non-instantiated generator object
            A waveform generator that produces a stream of floats from (-1, 1)
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `envelope` : :py:class:`pyglet.media.synthesis._Envelope`
            An optional Envelope to apply to the waveform.
    """

    def __init__(self, generator, duration, sample_rate=44800, envelope=None):
        self._generator = generator
        self._duration = duration
        self.audio_format = AudioFormat(channels=1, sample_size=16, sample_rate=sample_rate)
        self._envelope = envelope or FlatEnvelope(amplitude=1.0)
        self._envelope_generator = self._envelope.get_generator(sample_rate, duration)
        self._bytes_per_second = sample_rate * 2
        self._max_offset = int(self._bytes_per_second * duration) & 4294967294
        self._offset = 0

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        """Return `num_bytes` bytes of audio data."""
        num_bytes = min(num_bytes, self._max_offset - self._offset)
        if num_bytes <= 0:
            return None
        timestamp = self._offset / self._bytes_per_second
        duration = num_bytes / self._bytes_per_second
        self._offset += num_bytes
        samples = num_bytes >> 1
        generator = self._generator
        envelope = self._envelope_generator
        data = (int(next(generator) * next(envelope) * 32767) for _ in range(samples))
        data = _struct.pack(f'{samples}h', *data)
        return AudioData(data, num_bytes, timestamp, duration, [])

    def seek(self, timestamp):
        offset = int(timestamp * self._bytes_per_second)
        self._offset = min(max(offset, 0), self._max_offset) & 4294967294
        self._envelope_generator = self._envelope.get_generator(self.audio_format.sample_rate, self._duration)

    def is_precise(self) -> bool:
        return True