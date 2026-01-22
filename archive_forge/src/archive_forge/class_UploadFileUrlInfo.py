from __future__ import annotations
import io
from abc import abstractmethod
from typing import NamedTuple, Protocol, Sequence
from streamlit import util
from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
from streamlit.runtime.stats import CacheStatsProvider
class UploadFileUrlInfo(NamedTuple):
    """Information we provide for single file in get_upload_urls"""
    file_id: str
    upload_url: str
    delete_url: str