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
def _page_link(self, page: str, *, label: str | None=None, icon: str | None=None, help: str | None=None, disabled: bool=False, use_container_width: bool | None=None) -> DeltaGenerator:
    page_link_proto = PageLinkProto()
    page_link_proto.disabled = disabled
    if label is not None:
        page_link_proto.label = label
    if icon is not None:
        page_link_proto.icon = validate_emoji(icon)
    if help is not None:
        page_link_proto.help = dedent(help)
    if use_container_width is not None:
        page_link_proto.use_container_width = use_container_width
    if page.startswith('http://') or page.startswith('https://'):
        if label is None or label == '':
            raise StreamlitAPIException(f'The label param is required for external links used with st.page_link - please provide a label.')
        else:
            page_link_proto.page = page
            page_link_proto.external = True
            return self.dg._enqueue('page_link', page_link_proto)
    ctx = get_script_run_ctx()
    ctx_main_script = ''
    if ctx:
        ctx_main_script = ctx.main_script_path
    main_script_directory = get_main_script_directory(ctx_main_script)
    requested_page = os.path.realpath(normalize_path_join(main_script_directory, page))
    all_app_pages = source_util.get_pages(ctx_main_script).values()
    for page_data in all_app_pages:
        full_path = page_data['script_path']
        page_name = page_data['page_name']
        if requested_page == full_path:
            if label is None:
                page_link_proto.label = page_name.replace('_', ' ')
            page_link_proto.page_script_hash = page_data['page_script_hash']
            page_link_proto.page = page_name
            break
    if page_link_proto.page_script_hash == '':
        raise StreamlitAPIException(f'Could not find page: `{page}`. Must be the file path relative to the main script, from the directory: `{os.path.basename(main_script_directory)}`. Only the main app file and files in the `pages/` directory are supported.')
    return self.dg._enqueue('page_link', page_link_proto)