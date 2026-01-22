from asyncio import get_event_loop, iscoroutine
from functools import wraps
from inspect import signature

    Convert an asyncio coroutine into a function which, when called, is
    evaluted in an event loop, and the return value returned. This is intented
    to make it easy to write entry points into asyncio coroutines, which
    otherwise need to be explictly evaluted with an event loop's
    run_until_complete.

    If `loop` is given, it is used as the event loop to run the coro in. If it
    is None (the default), the loop is retreived using asyncio.get_event_loop.
    This call is defered until the decorated function is called, so that
    callers can install custom event loops or event loop policies after
    @autoasync is applied.

    If `forever` is True, the loop is run forever after the decorated coroutine
    is finished. Use this for servers created with asyncio.start_server and the
    like.

    If `pass_loop` is True, the event loop object is passed into the coroutine
    as the `loop` kwarg when the wrapper function is called. In this case, the
    wrapper function's __signature__ is updated to remove this parameter, so
    that autoparse can still be used on it without generating a parameter for
    `loop`.

    This coroutine can be called with ( @autoasync(...) ) or without
    ( @autoasync ) arguments.

    Examples:

    @autoasync
    def get_file(host, port):
        reader, writer = yield from asyncio.open_connection(host, port)
        data = reader.read()
        sys.stdout.write(data.decode())

    get_file(host, port)

    @autoasync(forever=True, pass_loop=True)
    def server(host, port, loop):
        yield_from loop.create_server(Proto, host, port)

    server('localhost', 8899)

    