from ast import literal_eval
import json
from ipywidgets import register, CallbackDispatcher, DOMWidget
from traitlets import Any, Bool, Int, Unicode
from ..data_utils.binary_transfer import data_buffer_serialization
from ._frontend import module_name, module_version
from .debounce import debounce
def _handle_custom_msgs(self, _, content, buffers=None):
    content = json.loads(content)
    event_type = content.get('type', '')
    if event_type == 'deck-hover-event':
        self._hover_handlers(self, content)
    elif event_type == 'deck-resize-event':
        self._resize_handlers(self, content)
    elif event_type == 'deck-view-state-change-event':
        self._view_state_handlers(self, content)
    elif event_type == 'deck-click-event':
        self._click_handlers(self, content)
    elif event_type == 'deck-drag-start-event':
        self._drag_start_handlers(self, content)
    elif event_type == 'deck-drag-event':
        self._drag_handlers(self, content)
    elif event_type == 'deck-drag-end-event':
        self._drag_end_handlers(self, content)