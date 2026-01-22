import sys
from collections import deque
from ctypes import (c_int, c_int32, c_uint8, c_char_p,
import pyglet
import pyglet.lib
from pyglet import image
from pyglet.util import asbytes, asstr
from . import MediaDecoder
from .base import AudioData, SourceInfo, StaticSource
from .base import StreamingSource, VideoFormat, AudioFormat
from .ffmpeg_lib import *
from ..exceptions import MediaFormatException
def _ffmpeg_decode_audio(self, packet, data_out, compensation_time):
    stream = self._audio_stream
    if stream.type != AVMEDIA_TYPE_AUDIO:
        raise FFmpegException('Trying to decode audio on a non-audio stream.')
    sent_result = avcodec.avcodec_send_packet(stream.codec_context, packet)
    if sent_result < 0:
        buf = create_string_buffer(128)
        avutil.av_strerror(sent_result, buf, 128)
        descr = buf.value
        raise FFmpegException('Error occurred sending packet to decoder. {}'.format(descr.decode()))
    receive_result = avcodec.avcodec_receive_frame(stream.codec_context, stream.frame)
    if receive_result < 0:
        buf = create_string_buffer(128)
        avutil.av_strerror(receive_result, buf, 128)
        descr = buf.value
        raise FFmpegException('Error occurred receiving frame. {}'.format(descr.decode()))
    plane_size = c_int(0)
    data_size = avutil.av_samples_get_buffer_size(byref(plane_size), stream.codec_context.contents.channels, stream.frame.contents.nb_samples, stream.codec_context.contents.sample_fmt, 1)
    if data_size < 0:
        raise FFmpegException('Error in av_samples_get_buffer_size')
    if len(self._audio_buffer) < data_size:
        raise FFmpegException('Output audio buffer is too small for current audio frame!')
    nb_samples = stream.frame.contents.nb_samples
    sample_rate = stream.codec_context.contents.sample_rate
    bytes_per_sample = avutil.av_get_bytes_per_sample(self.tgt_format)
    channels_out = min(2, self.audio_format.channels)
    wanted_nb_samples = nb_samples + compensation_time * sample_rate
    min_nb_samples = nb_samples * (100 - self.SAMPLE_CORRECTION_PERCENT_MAX) / 100
    max_nb_samples = nb_samples * (100 + self.SAMPLE_CORRECTION_PERCENT_MAX) / 100
    wanted_nb_samples = min(max(wanted_nb_samples, min_nb_samples), max_nb_samples)
    wanted_nb_samples = int(wanted_nb_samples)
    if wanted_nb_samples != nb_samples:
        res = swresample.swr_set_compensation(self.audio_convert_ctx, wanted_nb_samples - nb_samples, wanted_nb_samples)
        if res < 0:
            raise FFmpegException('swr_set_compensation failed.')
    data_in = stream.frame.contents.extended_data
    p_data_out = cast(data_out, POINTER(c_uint8))
    out_samples = swresample.swr_get_out_samples(self.audio_convert_ctx, nb_samples)
    total_samples_out = swresample.swr_convert(self.audio_convert_ctx, byref(p_data_out), out_samples, data_in, nb_samples)
    while True:
        offset = total_samples_out * channels_out * bytes_per_sample
        p_data_offset = cast(addressof(p_data_out.contents) + offset, POINTER(c_uint8))
        samples_out = swresample.swr_convert(self.audio_convert_ctx, byref(p_data_offset), out_samples - total_samples_out, None, 0)
        if samples_out == 0:
            break
        total_samples_out += samples_out
    size_out = total_samples_out * channels_out * bytes_per_sample
    return size_out