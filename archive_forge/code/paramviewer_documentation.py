from __future__ import annotations
from typing import Literal, TypedDict
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.events import Events

        Parameters:
            value: Expects value as a `dict[str, dict]`. The key in the outer dictionary is the parameter name, while the inner dictionary has keys "type", "description", and (optionally) "default" for each parameter.
        Returns:
            The same value.
        