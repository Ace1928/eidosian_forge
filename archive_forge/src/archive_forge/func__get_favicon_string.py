from __future__ import annotations
import random
from textwrap import dedent
from typing import TYPE_CHECKING, Final, Literal, Mapping, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements import image
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg as ForwardProto
from streamlit.proto.PageConfig_pb2 import PageConfig as PageConfigProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.string_util import is_emoji
from streamlit.url_util import is_url
from streamlit.util import lower_clean_dict_keys
def _get_favicon_string(page_icon: PageIcon) -> str:
    """Return the string to pass to the frontend to have it show
    the given PageIcon.

    If page_icon is a string that looks like an emoji (or an emoji shortcode),
    we return it as-is. Otherwise we use `image_to_url` to return a URL.

    (If `image_to_url` raises an error and page_icon is a string, return
    the unmodified page_icon string instead of re-raising the error.)
    """
    if page_icon == 'random':
        return get_random_emoji()
    if isinstance(page_icon, str) and is_emoji(page_icon):
        return page_icon
    try:
        return image.image_to_url(page_icon, width=-1, clamp=False, channels='RGB', output_format='auto', image_id='favicon')
    except Exception:
        if isinstance(page_icon, str):
            return page_icon
        raise