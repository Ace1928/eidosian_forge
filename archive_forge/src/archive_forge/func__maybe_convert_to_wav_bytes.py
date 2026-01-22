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
def _maybe_convert_to_wav_bytes(data: MediaData, sample_rate: int | None) -> MediaData:
    """Convert data to wav bytes if the data type is numpy array."""
    if type_util.is_type(data, 'numpy.ndarray') and sample_rate is not None:
        data = _make_wav(cast('npt.NDArray[Any]', data), sample_rate)
    return data