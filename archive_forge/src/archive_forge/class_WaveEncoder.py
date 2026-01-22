import wave
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
class WaveEncoder(MediaEncoder):

    def get_file_extensions(self):
        return ('.wav', '.wave', '.riff')

    def encode(self, source, filename, file):
        """Save the Source to disk as a standard RIFF Wave.

        A standard RIFF wave header will be added to the raw PCM
        audio data when it is saved to disk.

        :Parameters:
            `filename` : str
                The file name to save as.
            `file` : file-like object
                A file-like object, opened with mode 'wb'.

        """
        opened_file = None
        if file is None:
            file = open(filename, 'wb')
            opened_file = True
        source.seek(0)
        wave_writer = wave.open(file, mode='wb')
        wave_writer.setnchannels(source.audio_format.channels)
        wave_writer.setsampwidth(source.audio_format.bytes_per_sample)
        wave_writer.setframerate(source.audio_format.sample_rate)
        chunksize = source.audio_format.bytes_per_second
        audiodata = source.get_audio_data(chunksize)
        while audiodata:
            wave_writer.writeframes(audiodata.data)
            audiodata = source.get_audio_data(chunksize)
        else:
            wave_writer.close()
            if opened_file:
                file.close()