import enum
from tensorflow.python.util.tf_export import tf_export
class RemoteValueStatus(enum.Enum):
    """The status of a `RemoteValue` object.

  A `RemoteValue` object can have three states:
    1) not ready: no value, no non-retryable error and not aborted;
    2) aborted: i.e. the execution of function was aborted because of task
       failure, but can be retried;
    3) ready: i.e. has value or has non-tryable error;

  The initial state of a `RemoteValue` is "not ready". When its corresponding
  closure has
  been executed at least once, it will become aborted or ready. The state
  transitions are:
    1) not ready -> 2) aborted:
      when the corresponding closure is aborted due to worker failure, and the
      worker failure is not immediately handled.
    1) not ready -> 3) ready:
      when the corresponding closure has been executed successfully.
    2) aborted -> 3) ready:
      when the `RemoteValue` is rebuilt by rerunning the corresponding closure
      and the closure has been executed successfully.
    3) ready -> 2) aborted:
      when the corresponding closure had been executed successfully but later
      the corresponding remote worker failed. This is currently only implemented
      for resource `RemoteValue` like iterators.
  """
    NOT_READY = 'NOT_READY'
    ABORTED = 'ABORTED'
    READY = 'READY'