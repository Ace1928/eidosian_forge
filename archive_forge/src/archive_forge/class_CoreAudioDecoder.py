from ctypes import memmove, byref, c_uint32, sizeof, cast, c_void_p, create_string_buffer, POINTER, c_char, \
from pyglet.libs.darwin import cf, CFSTR
from pyglet.libs.darwin.coreaudio import kCFURLPOSIXPathStyle, AudioStreamBasicDescription, ca, ExtAudioFileRef, \
from pyglet.media import StreamingSource, StaticSource
from pyglet.media.codecs import AudioFormat, MediaDecoder, AudioData
class CoreAudioDecoder(MediaDecoder):

    def get_file_extensions(self):
        return ('.aac', '.ac3', '.aif', '.aiff', '.aifc', '.caf', '.mp3', '.mp4', '.m4a', '.snd', '.au', '.sd2', '.wav')

    def decode(self, filename, file, streaming=True):
        if streaming:
            return CoreAudioSource(filename, file)
        else:
            return StaticSource(CoreAudioSource(filename, file))