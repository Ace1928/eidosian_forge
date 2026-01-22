from __future__ import annotations
import hashlib
import io
import os
import re
from pathlib import Path
from streamlit import runtime
from streamlit.runtime import caching
def _handle_stream_data(stream: io.BytesIO) -> bytes:
    """Handles io.BytesIO data, converting SRT to VTT content if needed."""
    stream.seek(0)
    stream_data = stream.getvalue()
    return _srt_to_vtt(stream_data) if _is_srt(stream) else stream_data