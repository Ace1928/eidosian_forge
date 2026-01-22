from functools import reduce
class FallbackContext:
    """Context workaround.

    The built-in ``@contextmanager`` utility does not work well
    when wrapping other contexts, as the traceback is wrong when
    the wrapped context raises.

    This solves this problem and can be used instead of ``@contextmanager``
    in this example::

        @contextmanager
        def connection_or_default_connection(connection=None):
            if connection:
                # user already has a connection, shouldn't close
                # after use
                yield connection
            else:
                # must've new connection, and also close the connection
                # after the block returns
                with create_new_connection() as connection:
                    yield connection

    This wrapper can be used instead for the above like this::

        def connection_or_default_connection(connection=None):
            return FallbackContext(connection, create_new_connection)
    """

    def __init__(self, provided, fallback, *fb_args, **fb_kwargs):
        self.provided = provided
        self.fallback = fallback
        self.fb_args = fb_args
        self.fb_kwargs = fb_kwargs
        self._context = None

    def __enter__(self):
        if self.provided is not None:
            return self.provided
        context = self._context = self.fallback(*self.fb_args, **self.fb_kwargs).__enter__()
        return context

    def __exit__(self, *exc_info):
        if self._context is not None:
            return self._context.__exit__(*exc_info)