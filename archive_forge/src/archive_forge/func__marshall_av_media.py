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
def _marshall_av_media(coordinates: str, proto: AudioProto | VideoProto, data: MediaData, mimetype: str) -> None:
    """Fill audio or video proto based on contents of data.

    Given a string, check if it's a url; if so, send it out without modification.
    Otherwise assume strings are filenames and let any OS errors raise.

    Load data either from file or through bytes-processing methods into a
    MediaFile object.  Pack proto with generated Tornado-based URL.

    (When running in "raw" mode, we won't actually load data into the
    MediaFileManager, and we'll return an empty URL.)
    """
    if data is None:
        return
    data_or_filename: bytes | str
    if isinstance(data, (str, bytes)):
        data_or_filename = data
    elif isinstance(data, io.BytesIO):
        data.seek(0)
        data_or_filename = data.getvalue()
    elif isinstance(data, io.RawIOBase) or isinstance(data, io.BufferedReader):
        data.seek(0)
        read_data = data.read()
        if read_data is None:
            return
        else:
            data_or_filename = read_data
    elif type_util.is_type(data, 'numpy.ndarray'):
        data_or_filename = data.tobytes()
    else:
        raise RuntimeError('Invalid binary data format: %s' % type(data))
    if runtime.exists():
        file_url = runtime.get_instance().media_file_mgr.add(data_or_filename, mimetype, coordinates)
        caching.save_media_data(data_or_filename, mimetype, coordinates)
    else:
        file_url = ''
    proto.url = file_url