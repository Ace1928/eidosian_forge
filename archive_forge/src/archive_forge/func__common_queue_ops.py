import json
import zaqarclient.transport.errors as errors
def _common_queue_ops(operation, transport, request, name, callback=None):
    """Function for common operation

    This is a lower level call to get a single
    instance of queue.

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param name: Queue reference name.
    :type name: str
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    """
    request.operation = operation
    request.params['queue_name'] = name
    resp = transport.send(request)
    return resp.deserialized_content