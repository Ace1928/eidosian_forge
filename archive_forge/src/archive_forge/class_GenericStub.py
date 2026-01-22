import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class GenericStub(abc.ABC):
    """Affords RPC invocation via generic methods."""

    @abc.abstractmethod
    def blocking_unary_unary(self, group, method, request, timeout, metadata=None, with_call=False, protocol_options=None):
        """Invokes a unary-request-unary-response method.

        This method blocks until either returning the response value of the RPC
        (in the event of RPC completion) or raising an exception (in the event of
        RPC abortion).

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request: The request value for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          with_call: Whether or not to include return a Call for the RPC in addition
            to the response.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          The response value for the RPC, and a Call for the RPC if with_call was
            set to True at invocation.

        Raises:
          AbortionError: Indicating that the RPC was aborted.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def future_unary_unary(self, group, method, request, timeout, metadata=None, protocol_options=None):
        """Invokes a unary-request-unary-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request: The request value for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and a future.Future. In the
            event of RPC completion, the return Future's result value will be the
            response value of the RPC. In the event of RPC abortion, the returned
            Future's exception value will be an AbortionError.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def inline_unary_stream(self, group, method, request, timeout, metadata=None, protocol_options=None):
        """Invokes a unary-request-stream-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request: The request value for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and an iterator of response
            values. Drawing response values from the returned iterator may raise
            AbortionError indicating abortion of the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def blocking_stream_unary(self, group, method, request_iterator, timeout, metadata=None, with_call=False, protocol_options=None):
        """Invokes a stream-request-unary-response method.

        This method blocks until either returning the response value of the RPC
        (in the event of RPC completion) or raising an exception (in the event of
        RPC abortion).

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request_iterator: An iterator that yields request values for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          with_call: Whether or not to include return a Call for the RPC in addition
            to the response.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          The response value for the RPC, and a Call for the RPC if with_call was
            set to True at invocation.

        Raises:
          AbortionError: Indicating that the RPC was aborted.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def future_stream_unary(self, group, method, request_iterator, timeout, metadata=None, protocol_options=None):
        """Invokes a stream-request-unary-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request_iterator: An iterator that yields request values for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and a future.Future. In the
            event of RPC completion, the return Future's result value will be the
            response value of the RPC. In the event of RPC abortion, the returned
            Future's exception value will be an AbortionError.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def inline_stream_stream(self, group, method, request_iterator, timeout, metadata=None, protocol_options=None):
        """Invokes a stream-request-stream-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request_iterator: An iterator that yields request values for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and an iterator of response
            values. Drawing response values from the returned iterator may raise
            AbortionError indicating abortion of the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def event_unary_unary(self, group, method, request, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        """Event-driven invocation of a unary-request-unary-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request: The request value for the RPC.
          receiver: A ResponseReceiver to be passed the response data of the RPC.
          abortion_callback: A callback to be called and passed an Abortion value
            in the event of RPC abortion.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          A Call for the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def event_unary_stream(self, group, method, request, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        """Event-driven invocation of a unary-request-stream-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request: The request value for the RPC.
          receiver: A ResponseReceiver to be passed the response data of the RPC.
          abortion_callback: A callback to be called and passed an Abortion value
            in the event of RPC abortion.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          A Call for the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def event_stream_unary(self, group, method, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        """Event-driven invocation of a unary-request-unary-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          receiver: A ResponseReceiver to be passed the response data of the RPC.
          abortion_callback: A callback to be called and passed an Abortion value
            in the event of RPC abortion.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          A pair of a Call object for the RPC and a stream.Consumer to which the
            request values of the RPC should be passed.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def event_stream_stream(self, group, method, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        """Event-driven invocation of a unary-request-stream-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          receiver: A ResponseReceiver to be passed the response data of the RPC.
          abortion_callback: A callback to be called and passed an Abortion value
            in the event of RPC abortion.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          A pair of a Call object for the RPC and a stream.Consumer to which the
            request values of the RPC should be passed.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def unary_unary(self, group, method):
        """Creates a UnaryUnaryMultiCallable for a unary-unary method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.

        Returns:
          A UnaryUnaryMultiCallable value for the named unary-unary method.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def unary_stream(self, group, method):
        """Creates a UnaryStreamMultiCallable for a unary-stream method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.

        Returns:
          A UnaryStreamMultiCallable value for the name unary-stream method.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stream_unary(self, group, method):
        """Creates a StreamUnaryMultiCallable for a stream-unary method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.

        Returns:
          A StreamUnaryMultiCallable value for the named stream-unary method.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stream_stream(self, group, method):
        """Creates a StreamStreamMultiCallable for a stream-stream method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.

        Returns:
          A StreamStreamMultiCallable value for the named stream-stream method.
        """
        raise NotImplementedError()