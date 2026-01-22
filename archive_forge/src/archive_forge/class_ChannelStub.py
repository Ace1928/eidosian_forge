from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
class ChannelStub(grpc.Channel):
    """A testing stub for the grpc.Channel interface.

    This can be used to test any client that eventually uses a gRPC channel
    to communicate. By passing in a channel stub, you can configure which
    responses are returned and track which requests are made.

    For example:

    .. code-block:: python

        channel_stub = grpc_helpers.ChannelStub()
        client = FooClient(channel=channel_stub)

        channel_stub.GetFoo.response = foo_pb2.Foo(name='bar')

        foo = client.get_foo(labels=['baz'])

        assert foo.name == 'bar'
        assert channel_stub.GetFoo.requests[0].labels = ['baz']

    Each method on the stub can be accessed and configured on the channel.
    Here's some examples of various configurations:

    .. code-block:: python

        # Return a basic response:

        channel_stub.GetFoo.response = foo_pb2.Foo(name='bar')
        assert client.get_foo().name == 'bar'

        # Raise an exception:
        channel_stub.GetFoo.response = NotFound('...')

        with pytest.raises(NotFound):
            client.get_foo()

        # Use a sequence of responses:
        channel_stub.GetFoo.responses = iter([
            foo_pb2.Foo(name='bar'),
            foo_pb2.Foo(name='baz'),
        ])

        assert client.get_foo().name == 'bar'
        assert client.get_foo().name == 'baz'

        # Use a callable

        def on_get_foo(request):
            return foo_pb2.Foo(name='bar' + request.id)

        channel_stub.GetFoo.response = on_get_foo

        assert client.get_foo(id='123').name == 'bar123'
    """

    def __init__(self, responses=[]):
        self.requests = []
        'Sequence[Tuple[str, protobuf.Message]]: A list of all requests made\n        on this channel in order. The tuple is of method name, request\n        message.'
        self._method_stubs = {}

    def _stub_for_method(self, method):
        method = _simplify_method_name(method)
        self._method_stubs[method] = _CallableStub(method, self)
        return self._method_stubs[method]

    def __getattr__(self, key):
        try:
            return self._method_stubs[key]
        except KeyError:
            raise AttributeError

    def unary_unary(self, method, request_serializer=None, response_deserializer=None):
        """grpc.Channel.unary_unary implementation."""
        return self._stub_for_method(method)

    def unary_stream(self, method, request_serializer=None, response_deserializer=None):
        """grpc.Channel.unary_stream implementation."""
        return self._stub_for_method(method)

    def stream_unary(self, method, request_serializer=None, response_deserializer=None):
        """grpc.Channel.stream_unary implementation."""
        return self._stub_for_method(method)

    def stream_stream(self, method, request_serializer=None, response_deserializer=None):
        """grpc.Channel.stream_stream implementation."""
        return self._stub_for_method(method)

    def subscribe(self, callback, try_to_connect=False):
        """grpc.Channel.subscribe implementation."""
        pass

    def unsubscribe(self, callback):
        """grpc.Channel.unsubscribe implementation."""
        pass

    def close(self):
        """grpc.Channel.close implementation."""
        pass