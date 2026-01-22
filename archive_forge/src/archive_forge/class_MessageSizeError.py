from __future__ import annotations
import math
from datetime import timedelta
from typing import Any, Literal, overload
from streamlit import config
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed
class MessageSizeError(MarkdownFormattedException):
    """Exception raised when a websocket message is larger than the configured limit."""

    def __init__(self, failed_msg_str: Any):
        msg = self._get_message(failed_msg_str)
        super().__init__(msg)

    def _get_message(self, failed_msg_str: Any) -> str:
        return "\n**Data of size {message_size_mb:.1f} MB exceeds the message size limit of {message_size_limit_mb} MB.**\n\nThis is often caused by a large chart or dataframe. Please decrease the amount of data sent\nto the browser, or increase the limit by setting the config option `server.maxMessageSize`.\n[Click here to learn more about config options](https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options).\n\n_Note that increasing the limit may lead to long loading times and large memory consumption\nof the client's browser and the Streamlit server._\n".format(message_size_mb=len(failed_msg_str) / 1000000.0, message_size_limit_mb=get_max_message_size_bytes() / 1000000.0).strip('\n')