import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def triggered_id(self):
    """
        Returns the component id (str or dict) of the Input component that triggered the callback.

        Note - use `triggered_prop_ids` if you need both the component id and the prop that triggered the callback or if
        multiple Inputs triggered the callback.

        Example usage:
        `if "btn-1" == ctx.triggered_id:
            do_something()`

        """
    component_id = None
    if self.triggered:
        prop_id = self.triggered_prop_ids.first()
        component_id = self.triggered_prop_ids[prop_id]
    return component_id