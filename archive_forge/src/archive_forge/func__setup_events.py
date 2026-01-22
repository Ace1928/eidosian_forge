from __future__ import annotations
import inspect
from typing import AsyncGenerator, Callable, Literal, Union, cast
import anyio
from gradio_client.documentation import document
from gradio.blocks import Blocks
from gradio.components import (
from gradio.events import Dependency, on
from gradio.helpers import create_examples as Examples  # noqa: N812
from gradio.helpers import special_args
from gradio.layouts import Accordion, Group, Row
from gradio.routes import Request
from gradio.themes import ThemeClass as Theme
from gradio.utils import SyncToAsyncIterator, async_iteration, async_lambda
def _setup_events(self) -> None:
    submit_fn = self._stream_fn if self.is_generator else self._submit_fn
    submit_triggers = [self.textbox.submit, self.submit_btn.click] if self.submit_btn else [self.textbox.submit]
    submit_event = on(submit_triggers, self._clear_and_save_textbox, [self.textbox], [self.textbox, self.saved_input], show_api=False, queue=False).then(self._display_input, [self.saved_input, self.chatbot_state], [self.chatbot, self.chatbot_state], show_api=False, queue=False).then(submit_fn, [self.saved_input, self.chatbot_state] + self.additional_inputs, [self.chatbot, self.chatbot_state], show_api=False, concurrency_limit=cast(Union[int, Literal['default'], None], self.concurrency_limit))
    self._setup_stop_events(submit_triggers, submit_event)
    if self.retry_btn:
        retry_event = self.retry_btn.click(self._delete_prev_fn, [self.saved_input, self.chatbot_state], [self.chatbot, self.saved_input, self.chatbot_state], show_api=False, queue=False).then(self._display_input, [self.saved_input, self.chatbot_state], [self.chatbot, self.chatbot_state], show_api=False, queue=False).then(submit_fn, [self.saved_input, self.chatbot_state] + self.additional_inputs, [self.chatbot, self.chatbot_state], show_api=False, concurrency_limit=cast(Union[int, Literal['default'], None], self.concurrency_limit))
        self._setup_stop_events([self.retry_btn.click], retry_event)
    if self.undo_btn:
        self.undo_btn.click(self._delete_prev_fn, [self.saved_input, self.chatbot_state], [self.chatbot, self.saved_input, self.chatbot_state], show_api=False, queue=False).then(async_lambda(lambda x: x), [self.saved_input], [self.textbox], show_api=False, queue=False)
    if self.clear_btn:
        self.clear_btn.click(async_lambda(lambda: ([], [], None)), None, [self.chatbot, self.chatbot_state, self.saved_input], queue=False, show_api=False)