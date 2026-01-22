import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class MethodImplementation(abc.ABC):
    """A sum type that describes a method implementation.

    Attributes:
      cardinality: A cardinality.Cardinality value.
      style: A style.Service value.
      unary_unary_inline: The implementation of the method as a callable value
        that takes a request value and a ServicerContext object and returns a
        response value. Only non-None if cardinality is
        cardinality.Cardinality.UNARY_UNARY and style is style.Service.INLINE.
      unary_stream_inline: The implementation of the method as a callable value
        that takes a request value and a ServicerContext object and returns an
        iterator of response values. Only non-None if cardinality is
        cardinality.Cardinality.UNARY_STREAM and style is style.Service.INLINE.
      stream_unary_inline: The implementation of the method as a callable value
        that takes an iterator of request values and a ServicerContext object and
        returns a response value. Only non-None if cardinality is
        cardinality.Cardinality.STREAM_UNARY and style is style.Service.INLINE.
      stream_stream_inline: The implementation of the method as a callable value
        that takes an iterator of request values and a ServicerContext object and
        returns an iterator of response values. Only non-None if cardinality is
        cardinality.Cardinality.STREAM_STREAM and style is style.Service.INLINE.
      unary_unary_event: The implementation of the method as a callable value that
        takes a request value, a response callback to which to pass the response
        value of the RPC, and a ServicerContext. Only non-None if cardinality is
        cardinality.Cardinality.UNARY_UNARY and style is style.Service.EVENT.
      unary_stream_event: The implementation of the method as a callable value
        that takes a request value, a stream.Consumer to which to pass the
        response values of the RPC, and a ServicerContext. Only non-None if
        cardinality is cardinality.Cardinality.UNARY_STREAM and style is
        style.Service.EVENT.
      stream_unary_event: The implementation of the method as a callable value
        that takes a response callback to which to pass the response value of the
        RPC and a ServicerContext and returns a stream.Consumer to which the
        request values of the RPC should be passed. Only non-None if cardinality
        is cardinality.Cardinality.STREAM_UNARY and style is style.Service.EVENT.
      stream_stream_event: The implementation of the method as a callable value
        that takes a stream.Consumer to which to pass the response values of the
        RPC and a ServicerContext and returns a stream.Consumer to which the
        request values of the RPC should be passed. Only non-None if cardinality
        is cardinality.Cardinality.STREAM_STREAM and style is
        style.Service.EVENT.
    """