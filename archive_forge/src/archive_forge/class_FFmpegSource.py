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
class FFmpegSource(StreamingSource):
    SAMPLE_CORRECTION_PERCENT_MAX = 10
    MAX_QUEUE_SIZE = 100

    def __init__(self, filename, file=None):
        self._packet = None
        self._video_stream = None
        self._audio_stream = None
        self._stream_end = False
        self._file = None
        self._memory_file = None
        encoded_filename = filename.encode(sys.getfilesystemencoding())
        if file:
            self._file, self._memory_file = ffmpeg_open_memory_file(encoded_filename, file)
        else:
            self._file = ffmpeg_open_filename(encoded_filename)
        if not self._file:
            raise FFmpegException('Could not open "{0}"'.format(filename))
        self._video_stream_index = None
        self._audio_stream_index = None
        self._audio_format = None
        self.img_convert_ctx = POINTER(SwsContext)()
        self.audio_convert_ctx = POINTER(SwrContext)()
        file_info = ffmpeg_file_info(self._file)
        self.info = SourceInfo()
        self.info.title = file_info.title
        self.info.author = file_info.author
        self.info.copyright = file_info.copyright
        self.info.comment = file_info.comment
        self.info.album = file_info.album
        self.info.year = file_info.year
        self.info.track = file_info.track
        self.info.genre = file_info.genre
        for i in range(file_info.n_streams):
            info = ffmpeg_stream_info(self._file, i)
            if isinstance(info, StreamVideoInfo) and self._video_stream is None:
                stream = ffmpeg_open_stream(self._file, i)
                self.video_format = VideoFormat(width=info.width, height=info.height)
                if info.sample_aspect_num != 0:
                    self.video_format.sample_aspect = float(info.sample_aspect_num) / info.sample_aspect_den
                self.video_format.frame_rate = float(info.frame_rate_num) / info.frame_rate_den
                self._video_stream = stream
                self._video_stream_index = i
            elif isinstance(info, StreamAudioInfo) and info.sample_bits in (8, 16, 24) and (self._audio_stream is None):
                stream = ffmpeg_open_stream(self._file, i)
                self.audio_format = AudioFormat(channels=min(2, info.channels), sample_size=info.sample_bits, sample_rate=info.sample_rate)
                self._audio_stream = stream
                self._audio_stream_index = i
                channel_input = avutil.av_get_default_channel_layout(info.channels)
                channels_out = min(2, info.channels)
                channel_output = avutil.av_get_default_channel_layout(channels_out)
                sample_rate = stream.codec_context.contents.sample_rate
                sample_format = stream.codec_context.contents.sample_fmt
                if sample_format in (AV_SAMPLE_FMT_U8, AV_SAMPLE_FMT_U8P):
                    self.tgt_format = AV_SAMPLE_FMT_U8
                elif sample_format in (AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P):
                    self.tgt_format = AV_SAMPLE_FMT_S16
                elif sample_format in (AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_S32P):
                    self.tgt_format = AV_SAMPLE_FMT_S32
                elif sample_format in (AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP):
                    self.tgt_format = AV_SAMPLE_FMT_S16
                else:
                    raise FFmpegException('Audio format not supported.')
                self.audio_convert_ctx = swresample.swr_alloc_set_opts(None, channel_output, self.tgt_format, sample_rate, channel_input, sample_format, sample_rate, 0, None)
                if not self.audio_convert_ctx or swresample.swr_init(self.audio_convert_ctx) < 0:
                    swresample.swr_free(self.audio_convert_ctx)
                    raise FFmpegException('Cannot create sample rate converter.')
        self._packet = ffmpeg_init_packet()
        self._events = []
        self.audioq = deque()
        self._max_len_audioq = self.MAX_QUEUE_SIZE
        if self.audio_format:
            nbytes = ffmpeg_get_audio_buffer_size(self.audio_format)
            self._audio_buffer = (c_uint8 * nbytes)()
        self.videoq = deque()
        self._max_len_videoq = self.MAX_QUEUE_SIZE
        self.start_time = self._get_start_time()
        self._duration = timestamp_from_ffmpeg(file_info.duration)
        self._duration -= self.start_time
        self._fillq_scheduled = False
        self._fillq()
        if self.start_time > 0:
            self.seek(0.0)

    def __del__(self):
        if self._packet and ffmpeg_free_packet is not None:
            ffmpeg_free_packet(self._packet)
        if self._video_stream and swscale is not None:
            swscale.sws_freeContext(self.img_convert_ctx)
            ffmpeg_close_stream(self._video_stream)
        if self._audio_stream:
            swresample.swr_free(self.audio_convert_ctx)
            ffmpeg_close_stream(self._audio_stream)
        if self._file and ffmpeg_close_file is not None:
            ffmpeg_close_file(self._file)

    def seek(self, timestamp):
        if _debug:
            print('FFmpeg seek', timestamp)
        ffmpeg_seek_file(self._file, timestamp_to_ffmpeg(timestamp + self.start_time))
        del self._events[:]
        self._stream_end = False
        self._clear_video_audio_queues()
        self._fillq()
        if not self.audio_format:
            while len(self.videoq) > 1:
                if timestamp < self.videoq[1].timestamp:
                    break
                else:
                    self.get_next_video_frame(skip_empty_frame=False)
        elif not self.video_format:
            while len(self.audioq) > 1:
                if timestamp < self.audioq[1].timestamp:
                    break
                else:
                    self._get_audio_packet()
        else:
            while len(self.audioq) > 1 and len(self.videoq) > 1:
                audioq_is_first = self.audioq[0].timestamp < self.videoq[0].timestamp
                correct_audio_pos = timestamp < self.audioq[1].timestamp
                correct_video_pos = timestamp < self.videoq[1].timestamp
                if audioq_is_first and (not correct_audio_pos):
                    self._get_audio_packet()
                elif not correct_video_pos:
                    self.get_next_video_frame(skip_empty_frame=False)
                else:
                    break

    def _get_audio_packet(self):
        """Take an audio packet from the queue.

        This function will schedule its `_fillq` function to fill up
        the queues if space is available. Multiple calls to this method will
        only result in one scheduled call to `_fillq`.
        """
        audio_data = self.audioq.popleft()
        low_lvl = self._check_low_level()
        if not low_lvl and (not self._fillq_scheduled):
            pyglet.clock.schedule_once(lambda dt: self._fillq(), 0)
            self._fillq_scheduled = True
        return audio_data

    def _get_video_packet(self):
        """Take a video packet from the queue.

        This function will schedule its `_fillq` function to fill up
        the queues if space is available. Multiple calls to this method will
        only result in one scheduled call to `_fillq`.
        """
        if not self.videoq:
            return None
        video_packet = self.videoq.popleft()
        low_lvl = self._check_low_level()
        if not low_lvl and (not self._fillq_scheduled):
            pyglet.clock.schedule_once(lambda dt: self._fillq(), 0)
            self._fillq_scheduled = True
        return video_packet

    def _clear_video_audio_queues(self):
        """Empty both audio and video queues."""
        self.audioq.clear()
        self.videoq.clear()

    def _fillq(self):
        """Fill up both Audio and Video queues if space is available in both"""
        self._fillq_scheduled = False
        while len(self.audioq) < self._max_len_audioq and len(self.videoq) < self._max_len_videoq:
            if self._get_packet():
                self._process_packet()
            else:
                self._stream_end = True
                break

    def _check_low_level(self):
        """Check if both audio and video queues are getting very low.

        If one of them has less than 2 elements, we fill the queue immediately
        with new packets. We don't wait for a scheduled call because we need
        them immediately.

        This would normally happens only during seek operations where we
        consume many packets to find the correct timestamp.
        """
        if len(self.audioq) < 2 or len(self.videoq) < 2:
            if len(self.audioq) < self._max_len_audioq and len(self.videoq) < self._max_len_videoq:
                self._fillq()
            return True
        return False

    def _get_packet(self):
        return ffmpeg_read(self._file, self._packet)

    def _process_packet(self):
        """Process the packet that has been just read.

        Determines whether it's a video or audio packet and queue it in the
        appropriate queue.
        """
        timestamp = ffmpeg_get_packet_pts(self._file, self._packet)
        timestamp = timestamp_from_ffmpeg(timestamp)
        timestamp -= self.start_time
        if self._packet.contents.stream_index == self._video_stream_index:
            video_packet = VideoPacket(self._packet, timestamp)
            if _debug:
                print('Created and queued packet %d (%f)' % (video_packet.id, video_packet.timestamp))
            self.videoq.append(video_packet)
            return video_packet
        elif self.audio_format and self._packet.contents.stream_index == self._audio_stream_index:
            audio_packet = AudioPacket(self._packet, timestamp)
            self.audioq.append(audio_packet)
            return audio_packet

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        data = b''
        timestamp = duration = 0
        while len(data) < num_bytes:
            if not self.audioq:
                break
            audio_packet = self._get_audio_packet()
            buffer, timestamp, duration = self._decode_audio_packet(audio_packet, compensation_time)
            if not buffer:
                break
            data += buffer
        if not data and (not self.audioq):
            if not self._stream_end:
                if _debug:
                    print('Audio queue was starved by the audio driver.')
            return None
        audio_data = AudioData(data, len(data), timestamp, duration, [])
        while self._events and self._events[0].timestamp <= timestamp + duration:
            event = self._events.pop(0)
            if event.timestamp >= timestamp:
                event.timestamp -= timestamp
                audio_data.events.append(event)
        if _debug:
            print('get_audio_data returning ts {0} with events {1}'.format(audio_data.timestamp, audio_data.events))
            print('remaining events are', self._events)
        return audio_data

    def _decode_audio_packet(self, audio_packet, compensation_time):
        while True:
            try:
                size_out = self._ffmpeg_decode_audio(audio_packet.packet, self._audio_buffer, compensation_time)
            except FFmpegException:
                break
            if size_out <= 0:
                break
            buffer = create_string_buffer(size_out)
            memmove(buffer, self._audio_buffer, len(buffer))
            buffer = buffer.raw
            duration = float(len(buffer)) / self.audio_format.bytes_per_second
            timestamp = ffmpeg_get_frame_ts(self._audio_stream)
            timestamp = timestamp_from_ffmpeg(timestamp)
            return (buffer, timestamp, duration)
        return (None, 0, 0)

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

    def _decode_video_packet(self, video_packet):
        width = self.video_format.width
        height = self.video_format.height
        pitch = width * 4
        nbytes = pitch * height + FF_INPUT_BUFFER_PADDING_SIZE
        buffer = (c_uint8 * nbytes)()
        try:
            result = self._ffmpeg_decode_video(video_packet.packet, buffer)
        except FFmpegException:
            image_data = None
        else:
            image_data = image.ImageData(width, height, 'RGBA', buffer, pitch)
            timestamp = ffmpeg_get_frame_ts(self._video_stream)
            timestamp = timestamp_from_ffmpeg(timestamp)
            video_packet.timestamp = timestamp - self.start_time
        video_packet.image = image_data
        if _debug:
            print('Decoding video packet at timestamp', video_packet, video_packet.timestamp)

    def _ffmpeg_decode_video(self, packet, data_out):
        stream = self._video_stream
        rgba_ptrs = (POINTER(c_uint8) * 4)()
        rgba_stride = (c_int * 4)()
        width = stream.codec_context.contents.width
        height = stream.codec_context.contents.height
        if stream.type != AVMEDIA_TYPE_VIDEO:
            raise FFmpegException('Trying to decode video on a non-video stream.')
        sent_result = avcodec.avcodec_send_packet(stream.codec_context, packet)
        if sent_result < 0:
            buf = create_string_buffer(128)
            avutil.av_strerror(sent_result, buf, 128)
            descr = buf.value
            raise FFmpegException('Video: Error occurred sending packet to decoder. {}'.format(descr.decode()))
        receive_result = avcodec.avcodec_receive_frame(stream.codec_context, stream.frame)
        if receive_result < 0:
            buf = create_string_buffer(128)
            avutil.av_strerror(receive_result, buf, 128)
            descr = buf.value
            raise FFmpegException('Video: Error occurred receiving frame. {}'.format(descr.decode()))
        avutil.av_image_fill_arrays(rgba_ptrs, rgba_stride, data_out, AV_PIX_FMT_RGBA, width, height, 1)
        self.img_convert_ctx = swscale.sws_getCachedContext(self.img_convert_ctx, width, height, stream.codec_context.contents.pix_fmt, width, height, AV_PIX_FMT_RGBA, SWS_FAST_BILINEAR, None, None, None)
        swscale.sws_scale(self.img_convert_ctx, cast(stream.frame.contents.data, POINTER(POINTER(c_uint8))), stream.frame.contents.linesize, 0, height, rgba_ptrs, rgba_stride)
        return receive_result

    def get_next_video_timestamp(self):
        if not self.video_format:
            return
        ts = None
        if self.videoq:
            while True:
                try:
                    video_packet = self.videoq.popleft()
                except IndexError:
                    break
                if video_packet.image == 0:
                    self._decode_video_packet(video_packet)
                if video_packet.image is not None:
                    ts = video_packet.timestamp
                    self.videoq.appendleft(video_packet)
                    break
                self._get_video_packet()
        else:
            ts = None
        if _debug:
            print('Next video timestamp is', ts)
        return ts

    def get_next_video_frame(self, skip_empty_frame=True):
        if not self.video_format:
            return
        while True:
            video_packet = self._get_video_packet()
            if not video_packet:
                return None
            if video_packet.image == 0:
                self._decode_video_packet(video_packet)
            if video_packet.image is not None or not skip_empty_frame:
                break
        if _debug:
            print('Returning', video_packet)
        return video_packet.image

    def _get_start_time(self):

        def streams():
            format_context = self._file.context
            for idx in (self._video_stream_index, self._audio_stream_index):
                if idx is None:
                    continue
                stream = format_context.contents.streams[idx].contents
                yield stream

        def start_times(streams):
            yield 0
            for stream in streams:
                start = stream.start_time
                if start == AV_NOPTS_VALUE:
                    yield 0
                start_time = avutil.av_rescale_q(start, stream.time_base, AV_TIME_BASE_Q)
                start_time = timestamp_from_ffmpeg(start_time)
                yield start_time
        return max(start_times(streams()))

    @property
    def audio_format(self):
        return self._audio_format

    @audio_format.setter
    def audio_format(self, value):
        self._audio_format = value
        if value is None:
            self.audioq.clear()