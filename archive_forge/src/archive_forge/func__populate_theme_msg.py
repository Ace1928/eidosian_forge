from __future__ import annotations
import asyncio
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Callable, Final
import streamlit.elements.exception as exception_utils
from streamlit import config, runtime, source_util
from streamlit.case_converters import to_snake_case
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.Common_pb2 import FileURLs, FileURLsRequest
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.GitInfo_pb2 import GitInfo
from streamlit.proto.NewSession_pb2 import (
from streamlit.proto.PagesChanged_pb2 import PagesChanged
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.fragment import FragmentStorage, MemoryFragmentStorage
from streamlit.runtime.metrics_util import Installation
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.secrets import secrets_singleton
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.version import STREAMLIT_VERSION_STRING
from streamlit.watcher import LocalSourcesWatcher
def _populate_theme_msg(msg: CustomThemeConfig) -> None:
    enum_encoded_options = {'base', 'font'}
    theme_opts = config.get_options_for_section('theme')
    if not any(theme_opts.values()):
        return
    for option_name, option_val in theme_opts.items():
        if option_name not in enum_encoded_options and option_val is not None:
            setattr(msg, to_snake_case(option_name), option_val)
    base_map = {'light': msg.BaseTheme.LIGHT, 'dark': msg.BaseTheme.DARK}
    base = theme_opts['base']
    if base is not None:
        if base not in base_map:
            _LOGGER.warning(f'"{base}" is an invalid value for theme.base. Allowed values include {list(base_map.keys())}. Setting theme.base to "light".')
        else:
            msg.base = base_map[base]
    font_map = {'sans serif': msg.FontFamily.SANS_SERIF, 'serif': msg.FontFamily.SERIF, 'monospace': msg.FontFamily.MONOSPACE}
    font = theme_opts['font']
    if font is not None:
        if font not in font_map:
            _LOGGER.warning(f'"{font}" is an invalid value for theme.font. Allowed values include {list(font_map.keys())}. Setting theme.font to "sans serif".')
        else:
            msg.font = font_map[font]