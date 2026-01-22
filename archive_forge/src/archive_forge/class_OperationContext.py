import abc
import enum
import threading  # pylint: disable=unused-import
class OperationContext(abc.ABC):
    """Provides operation-related information and action."""

    @abc.abstractmethod
    def outcome(self):
        """Indicates the operation's outcome (or that the operation is ongoing).

        Returns:
          None if the operation is still active or the Outcome value for the
            operation if it has terminated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add_termination_callback(self, callback):
        """Adds a function to be called upon operation termination.

        Args:
          callback: A callable to be passed an Outcome value on operation
            termination.

        Returns:
          None if the operation has not yet terminated and the passed callback will
            later be called when it does terminate, or if the operation has already
            terminated an Outcome value describing the operation termination and the
            passed callback will not be called as a result of this method call.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def time_remaining(self):
        """Describes the length of allowed time remaining for the operation.

        Returns:
          A nonnegative float indicating the length of allowed time in seconds
          remaining for the operation to complete before it is considered to have
          timed out. Zero is returned if the operation has terminated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cancel(self):
        """Cancels the operation if the operation has not yet terminated."""
        raise NotImplementedError()

    @abc.abstractmethod
    def fail(self, exception):
        """Indicates that the operation has failed.

        Args:
          exception: An exception germane to the operation failure. May be None.
        """
        raise NotImplementedError()