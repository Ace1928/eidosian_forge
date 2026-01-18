from __future__ import annotations
import logging # isort:skip
from functools import wraps
from .bases import Property
def without_property_validation(input_function):
    """ Turn off property validation during update callbacks

    Example:
        .. code-block:: python

            @without_property_validation
            def update(attr, old, new):
                # do things without validation

    See Also:
        :class:`~bokeh.core.properties.validate`: context mangager for more fine-grained control

    """

    @wraps(input_function)
    def func(*args, **kwargs):
        with validate(False):
            return input_function(*args, **kwargs)
    return func