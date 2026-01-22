import traceback
from contextvars import copy_context
from . import BaseLongCallbackManager
from ..._callback_context import context_value
from ..._utils import AttributeDict
from ...exceptions import PreventUpdate
class DiskcacheLongCallbackManager(DiskcacheManager):
    """Deprecated: use `from dash import DiskcacheManager` instead."""