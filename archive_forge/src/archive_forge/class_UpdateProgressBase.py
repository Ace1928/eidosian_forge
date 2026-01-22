class UpdateProgressBase(object):
    """Keeps track on particular server update task.

    ``handler`` is a method of client plugin performing
    required update operation.
    Its first positional argument must be ``resource_id``
    and this method must be resilent to intermittent failures,
    returning ``True`` if API was successfully called, ``False`` otherwise.

    If result of API call is asynchronous, client plugin must have
    corresponding ``check_<handler>`` method.
    Its first positional argument must be ``resource_id``
    and it must return ``True`` or ``False`` indicating completeness
    of the update operation.

    For synchronous API calls,
    set ``complete`` attribute of this object to ``True``.

    ``[handler|checker]_extra`` arguments, if passed to constructor,
    should be dictionaries of

      {'args': tuple(), 'kwargs': dict()}

    structure and contain parameters with which corresponding ``handler`` and
    ``check_<handler>`` methods of client plugin must be called.
    ``args`` is automatically prepended with ``resource_id``.
    Missing ``args`` or ``kwargs`` are interpreted
    as empty tuple/dict respectively.
    Defaults are interpreted as both ``args`` and ``kwargs`` being empty.
    """

    def __init__(self, resource_id, handler, complete=False, called=False, handler_extra=None, checker_extra=None):
        self.complete = complete
        self.called = called
        self.handler = handler
        self.checker = 'check_%s' % handler
        hargs = handler_extra or {}
        self.handler_args = (resource_id,) + (hargs.get('args') or ())
        self.handler_kwargs = hargs.get('kwargs') or {}
        cargs = checker_extra or {}
        self.checker_args = (resource_id,) + (cargs.get('args') or ())
        self.checker_kwargs = cargs.get('kwargs') or {}