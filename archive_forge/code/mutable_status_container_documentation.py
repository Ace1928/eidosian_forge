from __future__ import annotations
import time
from types import TracebackType
from typing import Literal, cast
from typing_extensions import TypeAlias
from streamlit.cursor import Cursor
from streamlit.delta_generator import DeltaGenerator, _enqueue_message
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
Update the status container.

        Only specified arguments are updated. Container contents and unspecified
        arguments remain unchanged.

        Parameters
        ----------
        label : str or None
            A new label of the status container. If None, the label is not
            changed.

        expanded : bool or None
            The new expanded state of the status container. If None,
            the expanded state is not changed.

        state : "running", "complete", "error", or None
            The new state of the status container. This mainly changes the
            icon. If None, the state is not changed.
        