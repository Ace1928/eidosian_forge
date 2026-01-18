import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def triggered_prop_ids(self):
    """
        Returns a dictionary of all the Input props that changed and caused the callback to execute. It is empty when
        the callback is called on initial load, unless an Input prop got its value from another initial callback.
        Callbacks triggered by user actions typically have one item in triggered, unless the same action changes
        two props at once or the callback has several Input props that are all modified by another callback based
        on a single user action.

        triggered_prop_ids (dict):
        - keys (str) : the triggered "prop_id" composed of "component_id.component_property"
        - values (str or dict): the id of the component that triggered the callback. Will be the dict id for pattern matching callbacks

        Example - regular callback
        {"btn-1.n_clicks": "btn-1"}

        Example - pattern matching callbacks:
        {'{"index":0,"type":"filter-dropdown"}.value': {"index":0,"type":"filter-dropdown"}}

        Example usage:
        `if "btn-1.n_clicks" in ctx.triggered_prop_ids:
            do_something()`
        """
    triggered = getattr(_get_context_value(), 'triggered_inputs', [])
    ids = AttributeDict({})
    for item in triggered:
        component_id, _, _ = item['prop_id'].rpartition('.')
        ids[item['prop_id']] = component_id
        if component_id.startswith('{'):
            ids[item['prop_id']] = AttributeDict(json.loads(component_id))
    return ids