from __future__ import annotations
import io
import re
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Final, Union, cast
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import runtime, type_util, url_util
from streamlit.elements.lib.subtitle_utils import process_subtitle_data
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Audio_pb2 import Audio as AudioProto
from streamlit.proto.Video_pb2 import Video as VideoProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.runtime_util import duration_to_seconds
class MediaMixin:

    @gather_metrics('audio')
    def audio(self, data: MediaData, format: str='audio/wav', start_time: MediaTime=0, *, sample_rate: int | None=None, end_time: MediaTime | None=None, loop: bool=False) -> DeltaGenerator:
        """Display an audio player.

        Parameters
        ----------
        data : str, bytes, BytesIO, numpy.ndarray, or file
            Raw audio data, filename, or a URL pointing to the file to load.
            Raw data formats must include all necessary file headers to match the file
            format specified via ``format``.
            If ``data`` is a numpy array, it must either be a 1D array of the waveform
            or a 2D array of shape ``(num_channels, num_samples)`` with waveforms
            for all channels. See the default channel order at
            http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx

        format : str
            The mime type for the audio file. Defaults to ``"audio/wav"``.
            See https://tools.ietf.org/html/rfc4281 for more info.

        start_time: int, float, timedelta, str, or None
            The time from which the element should start playing. This can be
            one of the following:

            * ``None`` (default): The element plays from the beginning.
            * An``int`` or ``float`` specifying the time in seconds. ``float``
              values are rounded down to whole seconds.
            * A string specifying the time in a format supported by `Pandas'
              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,
              e.g. ``"2 minute"``, ``"20s"``, or ``"1m14s"``.
            * A ``timedelta`` object from `Python's built-in datetime library
              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,
              e.g. ``timedelta(seconds=70)``.
        sample_rate: int or None
            The sample rate of the audio data in samples per second. Only required if
            ``data`` is a numpy array.
        end_time: int, float, timedelta, str, or None
            The time at which the element should stop playing. This can be
            one of the following:

            * ``None`` (default): The element plays through to the end.
            * An ``int`` or ``float`` specifying the time in seconds. ``float``
              values are rounded down to whole seconds.
            * A string specifying the time in a format supported by `Pandas'
              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,
              e.g. ``"2 minute"``, ``"20s"``, or ``"1m14s"``.
            * A ``timedelta`` object from `Python's built-in datetime library
              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,
              e.g. ``timedelta(seconds=70)``.
        loop: bool
            Whether the audio should loop playback.

        Examples
        --------
        To display an audio player for a local file, specify the file's string
        path and format.

        >>> import streamlit as st
        >>>
        >>> st.audio("cat-purr.mp3", format="audio/mpeg", loop=True)

        .. output::
           https://doc-audio-purr.streamlit.app/
           height: 250px

        You can also pass ``bytes`` or ``numpy.ndarray`` objects to ``st.audio``.

        >>> import streamlit as st
        >>> import numpy as np
        >>>
        >>> audio_file = open("myaudio.ogg", "rb")
        >>> audio_bytes = audio_file.read()
        >>>
        >>> st.audio(audio_bytes, format="audio/ogg")
        >>>
        >>> sample_rate = 44100  # 44100 samples per second
        >>> seconds = 2  # Note duration of 2 seconds
        >>> frequency_la = 440  # Our played note will be 440 Hz
        >>> # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
        >>> t = np.linspace(0, seconds, seconds * sample_rate, False)
        >>> # Generate a 440 Hz sine wave
        >>> note_la = np.sin(frequency_la * t * 2 * np.pi)
        >>>
        >>> st.audio(note_la, sample_rate=sample_rate)

        .. output::
           https://doc-audio.streamlit.app/
           height: 865px

        """
        start_time, end_time = _parse_start_time_end_time(start_time, end_time)
        audio_proto = AudioProto()
        coordinates = self.dg._get_delta_path_str()
        is_data_numpy_array = type_util.is_type(data, 'numpy.ndarray')
        if is_data_numpy_array and sample_rate is None:
            raise StreamlitAPIException('`sample_rate` must be specified when `data` is a numpy array.')
        if not is_data_numpy_array and sample_rate is not None:
            st.warning('Warning: `sample_rate` will be ignored since data is not a numpy array.')
        marshall_audio(coordinates, audio_proto, data, format, start_time, sample_rate, end_time, loop)
        return self.dg._enqueue('audio', audio_proto)

    @gather_metrics('video')
    def video(self, data: MediaData, format: str='video/mp4', start_time: MediaTime=0, *, subtitles: SubtitleData=None, end_time: MediaTime | None=None, loop: bool=False) -> DeltaGenerator:
        """Display a video player.

        Parameters
        ----------
        data : str, bytes, io.BytesIO, numpy.ndarray, or file
            Raw video data, filename, or URL pointing to a video to load.
            Includes support for YouTube URLs.
            Numpy arrays and raw data formats must include all necessary file
            headers to match specified file format.

        format : str
            The mime type for the video file. Defaults to ``"video/mp4"``.
            See https://tools.ietf.org/html/rfc4281 for more info.

        start_time: int, float, timedelta, str, or None
            The time from which the element should start playing. This can be
            one of the following:

            * ``None`` (default): The element plays from the beginning.
            * An``int`` or ``float`` specifying the time in seconds. ``float``
              values are rounded down to whole seconds.
            * A string specifying the time in a format supported by `Pandas'
              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,
              e.g. ``"2 minute"``, ``"20s"``, or ``"1m14s"``.
            * A ``timedelta`` object from `Python's built-in datetime library
              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,
              e.g. ``timedelta(seconds=70)``.
        subtitles: str, bytes, Path, io.BytesIO, or dict
            Optional subtitle data for the video, supporting several input types:

            * ``None`` (default): No subtitles.

            * A string, bytes, or Path: File path to a subtitle file in ``.vtt`` or ``.srt`` formats, or
              the raw content of subtitles conforming to these formats.
              If providing raw content, the string must adhere to the WebVTT or SRT
              format specifications.

            * io.BytesIO: A BytesIO stream that contains valid ``.vtt`` or ``.srt``
              formatted subtitle data.

            * A dictionary: Pairs of labels and file paths or raw subtitle content in
              ``.vtt`` or ``.srt`` formats to enable multiple subtitle tracks.
              The label will be shown in the video player. Example:
              ``{"English": "path/to/english.vtt", "French": "path/to/french.srt"}``

            When provided, subtitles are displayed by default. For multiple
            tracks, the first one is displayed by default. If you don't want any
            subtitles displayed by default, use an empty string for the value
            in a dictrionary's first pair: ``{"None": "", "English": "path/to/english.vtt"}``

            Not supported for YouTube videos.
        end_time: int, float, timedelta, str, or None
            The time at which the element should stop playing. This can be
            one of the following:

            * ``None`` (default): The element plays through to the end.
            * An ``int`` or ``float`` specifying the time in seconds. ``float``
              values are rounded down to whole seconds.
            * A string specifying the time in a format supported by `Pandas'
              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,
              e.g. ``"2 minute"``, ``"20s"``, or ``"1m14s"``.
            * A ``timedelta`` object from `Python's built-in datetime library
              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,
              e.g. ``timedelta(seconds=70)``.
        loop: bool
            Whether the video should loop playback.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> video_file = open('myvideo.mp4', 'rb')
        >>> video_bytes = video_file.read()
        >>>
        >>> st.video(video_bytes)

        .. output::
           https://doc-video.streamlit.app/
           height: 700px

        When you include subtitles, they will be turned on by default. A viewer
        can turn off the subtitles (or captions) from the browser's default video
        control menu, usually located in the lower-right corner of the video.

        Here is a simple VTT file (``subtitles.vtt``):

        >>> WEBVTT
        >>>
        >>> 0:00:01.000 --> 0:00:02.000
        >>> Look!
        >>>
        >>> 0:00:03.000 --> 0:00:05.000
        >>> Look at the pretty stars!

        If the above VTT file lives in the same directory as your app, you can
        add subtitles like so:

        >>> import streamlit as st
        >>>
        >>> VIDEO_URL = "https://example.com/not-youtube.mp4"
        >>> st.video(VIDEO_URL, subtitles="subtitles.vtt")

        .. output::
           https://doc-video-subtitles.streamlit.app/
           height: 700px

        See additional examples of supported subtitle input types in our
        `video subtitles feature demo <https://doc-video-subtitle-inputs.streamlit.app/>`_.

        .. note::
           Some videos may not display if they are encoded using MP4V (which is an export option in OpenCV), as this codec is
           not widely supported by browsers. Converting your video to H.264 will allow the video to be displayed in Streamlit.
           See this `StackOverflow post <https://stackoverflow.com/a/49535220/2394542>`_ or this
           `Streamlit forum post <https://discuss.streamlit.io/t/st-video-doesnt-show-opencv-generated-mp4/3193/2>`_
           for more information.

        """
        start_time, end_time = _parse_start_time_end_time(start_time, end_time)
        video_proto = VideoProto()
        coordinates = self.dg._get_delta_path_str()
        marshall_video(coordinates, video_proto, data, format, start_time, subtitles, end_time, loop)
        return self.dg._enqueue('video', video_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)