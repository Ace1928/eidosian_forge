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
def _setup_api(self) -> None:
    api_fn = self._api_stream_fn if self.is_generator else self._api_submit_fn
    self.fake_api_btn.click(api_fn, [self.textbox, self.chatbot_state] + self.additional_inputs, [self.textbox, self.chatbot_state], api_name='chat', concurrency_limit=cast(Union[int, Literal['default'], None], self.concurrency_limit))