from __future__ import annotations
import urllib.parse as parse
from typing import Any
from streamlit import util
from streamlit.constants import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
def _ensure_no_embed_params(query_params: dict[str, list[str] | str], query_string: str) -> str:
    """Ensures there are no embed params set (raises StreamlitAPIException) if there is a try,
    also makes sure old param values in query_string are preserved. Returns query_string : str.
    """
    query_params_without_embed = util.exclude_keys_in_dict(query_params, keys_to_exclude=EMBED_QUERY_PARAMS_KEYS)
    if query_params != query_params_without_embed:
        raise StreamlitAPIException('Query param embed and embed_options (case-insensitive) cannot be set using set_query_params method.')
    all_current_params = parse.parse_qs(query_string, keep_blank_values=True)
    current_embed_params = parse.urlencode({EMBED_QUERY_PARAM: [param for param in util.extract_key_query_params(all_current_params, param_key=EMBED_QUERY_PARAM)], EMBED_OPTIONS_QUERY_PARAM: [param for param in util.extract_key_query_params(all_current_params, param_key=EMBED_OPTIONS_QUERY_PARAM)]}, doseq=True)
    query_string = parse.urlencode(query_params, doseq=True)
    if query_string:
        separator = '&' if current_embed_params else ''
        return separator.join([query_string, current_embed_params])
    return current_embed_params