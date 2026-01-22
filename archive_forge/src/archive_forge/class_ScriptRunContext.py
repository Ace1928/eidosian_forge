from __future__ import annotations
import collections
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Counter, Dict, Final, Union
from urllib import parse
from typing_extensions import TypeAlias
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Command
from streamlit.runtime.scriptrunner.script_requests import ScriptRequests
from streamlit.runtime.state import SafeSessionState
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
import streamlit
@dataclass
class ScriptRunContext:
    """A context object that contains data for a "script run" - that is,
    data that's scoped to a single ScriptRunner execution (and therefore also
    scoped to a single connected "session").

    ScriptRunContext is used internally by virtually every `st.foo()` function.
    It is accessed only from the script thread that's created by ScriptRunner,
    or from app-created helper threads that have been "attached" to the
    ScriptRunContext via `add_script_run_ctx`.

    Streamlit code typically retrieves the active ScriptRunContext via the
    `get_script_run_ctx` function.
    """
    session_id: str
    _enqueue: Callable[[ForwardMsg], None]
    query_string: str
    session_state: SafeSessionState
    uploaded_file_mgr: UploadedFileManager
    main_script_path: str
    page_script_hash: str
    user_info: UserInfo
    fragment_storage: 'FragmentStorage'
    gather_usage_stats: bool = False
    command_tracking_deactivated: bool = False
    tracked_commands: list[Command] = field(default_factory=list)
    tracked_commands_counter: Counter[str] = field(default_factory=collections.Counter)
    _set_page_config_allowed: bool = True
    _has_script_started: bool = False
    widget_ids_this_run: set[str] = field(default_factory=set)
    widget_user_keys_this_run: set[str] = field(default_factory=set)
    form_ids_this_run: set[str] = field(default_factory=set)
    cursors: dict[int, 'streamlit.cursor.RunningCursor'] = field(default_factory=dict)
    script_requests: ScriptRequests | None = None
    current_fragment_id: str | None = None
    fragment_ids_this_run: set[str] | None = None
    _experimental_query_params_used = False
    _production_query_params_used = False

    def reset(self, query_string: str='', page_script_hash: str='', fragment_ids_this_run: set[str] | None=None) -> None:
        self.cursors = {}
        self.widget_ids_this_run = set()
        self.widget_user_keys_this_run = set()
        self.form_ids_this_run = set()
        self.query_string = query_string
        self.page_script_hash = page_script_hash
        self._set_page_config_allowed = True
        self._has_script_started = False
        self.command_tracking_deactivated: bool = False
        self.tracked_commands = []
        self.tracked_commands_counter = collections.Counter()
        self.current_fragment_id = None
        self.fragment_ids_this_run = fragment_ids_this_run
        parsed_query_params = parse.parse_qs(query_string, keep_blank_values=True)
        with self.session_state.query_params() as qp:
            qp.clear_with_no_forward_msg()
            for key, val in parsed_query_params.items():
                if len(val) == 0:
                    qp.set_with_no_forward_msg(key, val='')
                elif len(val) == 1:
                    qp.set_with_no_forward_msg(key, val=val[-1])
                else:
                    qp.set_with_no_forward_msg(key, val)

    def on_script_start(self) -> None:
        self._has_script_started = True

    def enqueue(self, msg: ForwardMsg) -> None:
        """Enqueue a ForwardMsg for this context's session."""
        if msg.HasField('page_config_changed') and (not self._set_page_config_allowed):
            raise StreamlitAPIException('`set_page_config()` can only be called once per app page, ' + 'and must be called as the first Streamlit command in your script.\n\n' + 'For more information refer to the [docs]' + '(https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config).')
        if msg.HasField('page_config_changed') or (msg.HasField('delta') and self._has_script_started):
            self._set_page_config_allowed = False
        self._enqueue(msg)

    def ensure_single_query_api_used(self):
        if self._experimental_query_params_used and self._production_query_params_used:
            raise StreamlitAPIException('Using `st.query_params` together with either `st.experimental_get_query_params` ' + 'or `st.experimental_set_query_params` is not supported. Please convert your app ' + 'to only use `st.query_params`')

    def mark_experimental_query_params_used(self):
        self._experimental_query_params_used = True
        self.ensure_single_query_api_used()

    def mark_production_query_params_used(self):
        self._production_query_params_used = True
        self.ensure_single_query_api_used()