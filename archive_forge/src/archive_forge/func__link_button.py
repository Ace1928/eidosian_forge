from __future__ import annotations
import io
import os
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, TextIO, Union, cast
from typing_extensions import TypeAlias
from streamlit import runtime, source_util
from streamlit.elements.form import current_form_id, is_in_form
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.file_util import get_main_script_directory, normalize_path_join
from streamlit.proto.Button_pb2 import Button as ButtonProto
from streamlit.proto.DownloadButton_pb2 import DownloadButton as DownloadButtonProto
from streamlit.proto.LinkButton_pb2 import LinkButton as LinkButtonProto
from streamlit.proto.PageLink_pb2 import PageLink as PageLinkProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.string_util import validate_emoji
from streamlit.type_util import Key, to_key
def _link_button(self, label: str, url: str, help: str | None, *, type: Literal['primary', 'secondary']='secondary', disabled: bool=False, use_container_width: bool=False) -> DeltaGenerator:
    link_button_proto = LinkButtonProto()
    link_button_proto.label = label
    link_button_proto.url = url
    link_button_proto.type = type
    link_button_proto.use_container_width = use_container_width
    link_button_proto.disabled = disabled
    if help is not None:
        link_button_proto.help = dedent(help)
    return self.dg._enqueue('link_button', link_button_proto)