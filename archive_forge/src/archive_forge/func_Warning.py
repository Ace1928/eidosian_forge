from __future__ import annotations
import ast
import csv
import inspect
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional
import numpy as np
import PIL
import PIL.Image
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import components, oauth, processing_utils, routes, utils, wasm_utils
from gradio.context import Context, LocalContext
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import EventData
from gradio.exceptions import Error
from gradio.flagging import CSVLogger
@document(documentation_group='modals')
def Warning(message: str='Warning issued.'):
    """
    This function allows you to pass custom warning messages to the user. You can do so simply by writing `gr.Warning('message here')` in your function, and when that line is executed the custom message will appear in a modal on the demo. The modal is yellow by default and has the heading: "Warning." Queue must be enabled for this behavior; otherwise, the warning will be printed to the console using the `warnings` library.
    Demos: blocks_chained_events
    Parameters:
        message: The warning message to be displayed to the user.
    Example:
        import gradio as gr
        def hello_world():
            gr.Warning('This is a warning message.')
            return "hello world"
        with gr.Blocks() as demo:
            md = gr.Markdown()
            demo.load(hello_world, inputs=None, outputs=[md])
        demo.queue().launch()
    """
    log_message(message, level='warning')