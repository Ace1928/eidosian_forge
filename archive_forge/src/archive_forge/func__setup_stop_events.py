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
def _setup_stop_events(self, event_triggers: list[Callable], event_to_cancel: Dependency) -> None:
    if self.stop_btn and self.is_generator:
        if self.submit_btn:
            for event_trigger in event_triggers:
                event_trigger(async_lambda(lambda: (Button(visible=False), Button(visible=True))), None, [self.submit_btn, self.stop_btn], show_api=False, queue=False)
            event_to_cancel.then(async_lambda(lambda: (Button(visible=True), Button(visible=False))), None, [self.submit_btn, self.stop_btn], show_api=False, queue=False)
        else:
            for event_trigger in event_triggers:
                event_trigger(async_lambda(lambda: Button(visible=True)), None, [self.stop_btn], show_api=False, queue=False)
            event_to_cancel.then(async_lambda(lambda: Button(visible=False)), None, [self.stop_btn], show_api=False, queue=False)
        self.stop_btn.click(None, None, None, cancels=event_to_cancel, show_api=False)