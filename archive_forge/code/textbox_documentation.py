from __future__ import annotations
from typing import Any, Callable, Literal
from gradio_client.documentation import document
from gradio.components.base import FormComponent
from gradio.events import Events

        Parameters:
            value: Expects a {str} returned from function and sets textarea value to it.
        Returns:
            The value to display in the textarea.
        