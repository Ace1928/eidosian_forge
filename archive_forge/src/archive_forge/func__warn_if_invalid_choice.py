from __future__ import annotations
import warnings
from typing import Any, Callable, Literal
from gradio_client.documentation import document
from gradio.components.base import FormComponent
from gradio.events import Events
def _warn_if_invalid_choice(self, value):
    if self.allow_custom_value or value in [value for _, value in self.choices]:
        return
    warnings.warn(f'The value passed into gr.Dropdown() is not in the list of choices. Please update the list of choices to include: {value} or set allow_custom_value=True.')