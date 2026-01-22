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
def _clear_and_save_textbox(self, message: str) -> tuple[str | dict, str]:
    if self.multimodal:
        return ({'text': '', 'files': []}, message)
    else:
        return ('', message)