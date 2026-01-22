import collections
import hashlib
from functools import wraps
import flask
from .dependencies import (
from .exceptions import (
from ._grouping import (
from ._utils import (
from . import _validate
from .long_callback.managers import BaseLongCallbackManager
from ._callback_context import context_value

    Normally used as a decorator, `@dash.callback` provides a server-side
    callback relating the values of one or more `Output` items to one or
    more `Input` items which will trigger the callback when they change,
    and optionally `State` items which provide additional information but
    do not trigger the callback directly.

    `@dash.callback` is an alternative to `@app.callback` (where `app = dash.Dash()`)
    introduced in Dash 2.0.
    It allows you to register callbacks without defining or importing the `app`
    object. The call signature is identical and it can be used instead of `app.callback`
    in all cases.

    The last, optional argument `prevent_initial_call` causes the callback
    not to fire when its outputs are first added to the page. Defaults to
    `False` and unlike `app.callback` is not configurable at the app level.

    :Keyword Arguments:
        :param background:
            Mark the callback as a long callback to execute in a manager for
            callbacks that take a long time without locking up the Dash app
            or timing out.
        :param manager:
            A long callback manager instance. Currently, an instance of one of
            `DiskcacheManager` or `CeleryManager`.
            Defaults to the `background_callback_manager` instance provided to the
            `dash.Dash constructor`.
            - A diskcache manager (`DiskcacheManager`) that runs callback
              logic in a separate process and stores the results to disk using the
              diskcache library. This is the easiest backend to use for local
              development.
            - A Celery manager (`CeleryManager`) that runs callback logic
              in a celery worker and returns results to the Dash app through a Celery
              broker like RabbitMQ or Redis.
        :param running:
            A list of 3-element tuples. The first element of each tuple should be
            an `Output` dependency object referencing a property of a component in
            the app layout. The second element is the value that the property
            should be set to while the callback is running, and the third element
            is the value the property should be set to when the callback completes.
        :param cancel:
            A list of `Input` dependency objects that reference a property of a
            component in the app's layout.  When the value of this property changes
            while a callback is running, the callback is canceled.
            Note that the value of the property is not significant, any change in
            value will result in the cancellation of the running job (if any).
        :param progress:
            An `Output` dependency grouping that references properties of
            components in the app's layout. When provided, the decorated function
            will be called with an extra argument as the first argument to the
            function.  This argument, is a function handle that the decorated
            function should call in order to provide updates to the app on its
            current progress. This function accepts a single argument, which
            correspond to the grouping of properties specified in the provided
            `Output` dependency grouping
        :param progress_default:
            A grouping of values that should be assigned to the components
            specified by the `progress` argument when the callback is not in
            progress. If `progress_default` is not provided, all the dependency
            properties specified in `progress` will be set to `None` when the
            callback is not running.
        :param cache_args_to_ignore:
            Arguments to ignore when caching is enabled. If callback is configured
            with keyword arguments (Input/State provided in a dict),
            this should be a list of argument names as strings. Otherwise,
            this should be a list of argument indices as integers.
        :param interval:
            Time to wait between the long callback update requests.
    