from __future__ import annotations
from typing import (
import param
from ..models.chatarea_input import (
from ..widgets import TextAreaInput as _PnTextAreaInput
from bokeh.model import Model
class ChatAreaInput(_PnTextAreaInput):
    """
    The `ChatAreaInput` allows entering any multiline string using a text input
    box, with the ability to press enter to submit the message.

    Unlike TextAreaInput, the `ChatAreaInput` defaults to auto_grow=True and
    max_rows=10, and the value is not synced to the server until the enter key
    is pressed so bind on `value_input` if you need to access the existing value.

    Lines are joined with the newline character `\\n`.

    Reference: https://panel.holoviz.org/reference/chat/ChatAreaInput.html

    :Example:

    >>> ChatAreaInput(max_rows=10)
    """
    auto_grow = param.Boolean(default=True, doc='\n        Whether the text area should automatically grow vertically to\n        accommodate the current text.')
    disabled_enter = param.Boolean(default=False, doc='If True, the enter key will not submit the message (clear the value).')
    rows = param.Integer(default=1, doc='\n        Number of rows in the text input field.')
    max_rows = param.Integer(default=10, doc='\n        When combined with auto_grow this determines the maximum number\n        of rows the input area can grow.')
    resizable = param.ObjectSelector(default='height', objects=['both', 'width', 'height', False], doc='\n        Whether the layout is interactively resizable,\n        and if so in which dimensions: `width`, `height`, or `both`.\n        Can only be set during initialization.')
    _widget_type: ClassVar[Type[Model]] = _bkChatAreaInput
    _rename: ClassVar[Mapping[str, str | None]] = {'value': None, **_PnTextAreaInput._rename}

    def _get_properties(self, doc: Document) -> Dict[str, Any]:
        props = super()._get_properties(doc)
        props.update({'value_input': self.value, 'value': self.value})
        return props

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        model = super()._get_model(doc, root, parent, comm)
        self._register_events('chat_message_event', model=model, doc=doc, comm=comm)
        return model

    def _process_event(self, event: ChatMessageEvent) -> None:
        """
        Clear value on shift enter key down.
        """
        self.value = event.value
        with param.discard_events(self):
            self.value = ''