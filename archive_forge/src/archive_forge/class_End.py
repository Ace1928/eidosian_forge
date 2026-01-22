import abc
import enum
import threading  # pylint: disable=unused-import
class End(abc.ABC):
    """Common type for entry-point objects on both sides of an operation."""

    @abc.abstractmethod
    def start(self):
        """Starts this object's service of operations."""
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self, grace):
        """Stops this object's service of operations.

        This object will refuse service of new operations as soon as this method is
        called but operations under way at the time of the call may be given a
        grace period during which they are allowed to finish.

        Args:
          grace: A duration of time in seconds to allow ongoing operations to
            terminate before being forcefully terminated by the stopping of this
            End. May be zero to terminate all ongoing operations and immediately
            stop.

        Returns:
          A threading.Event that will be set to indicate all operations having
            terminated and this End having completely stopped. The returned event
            may not be set until after the full grace period (if some ongoing
            operation continues for the full length of the period) or it may be set
            much sooner (if for example this End had no operations in progress at
            the time its stop method was called).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def operate(self, group, method, subscription, timeout, initial_metadata=None, payload=None, completion=None, protocol_options=None):
        """Commences an operation.

        Args:
          group: The group identifier of the invoked operation.
          method: The method identifier of the invoked operation.
          subscription: A Subscription to which the results of the operation will be
            passed.
          timeout: A length of time in seconds to allow for the operation.
          initial_metadata: An initial metadata value to be sent to the other side
            of the operation. May be None if the initial metadata will be later
            passed via the returned operator or if there will be no initial metadata
            passed at all.
          payload: An initial payload for the operation.
          completion: A Completion value indicating the end of transmission to the
            other side of the operation.
          protocol_options: A value specified by the provider of a Base interface
            implementation affording custom state and behavior.

        Returns:
          A pair of objects affording information about the operation and action
            continuing the operation. The first element of the returned pair is an
            OperationContext for the operation and the second element of the
            returned pair is an Operator to which operation values not passed in
            this call should later be passed.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def operation_stats(self):
        """Reports the number of terminated operations broken down by outcome.

        Returns:
          A dictionary from Outcome.Kind value to an integer identifying the number
            of operations that terminated with that outcome kind.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add_idle_action(self, action):
        """Adds an action to be called when this End has no ongoing operations.

        Args:
          action: A callable that accepts no arguments.
        """
        raise NotImplementedError()