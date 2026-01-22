from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, cast
from streamlit import runtime
from streamlit.elements.form import is_in_form
from streamlit.elements.image import AtomicImage, WidthBehaviour, image_to_url
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.ChatInput_pb2 import ChatInput as ChatInputProto
from streamlit.proto.Common_pb2 import StringTriggerValue as StringTriggerValueProto
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.string_util import is_emoji
from streamlit.type_util import Key, to_key
@dataclass
class ChatInputSerde:

    def deserialize(self, ui_value: StringTriggerValueProto | None, widget_id: str='') -> str | None:
        if ui_value is None or not ui_value.HasField('data'):
            return None
        return ui_value.data

    def serialize(self, v: str | None) -> StringTriggerValueProto:
        return StringTriggerValueProto(data=v)