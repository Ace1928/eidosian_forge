from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class UpdateConfig(dict):
    """

    Used to specify the way container updates should be performed by a service.

    Args:

        parallelism (int): Maximum number of tasks to be updated in one
          iteration (0 means unlimited parallelism). Default: 0.
        delay (int): Amount of time between updates, in nanoseconds.
        failure_action (string): Action to take if an updated task fails to
          run, or stops running during the update. Acceptable values are
          ``continue``, ``pause``, as well as ``rollback`` since API v1.28.
          Default: ``continue``
        monitor (int): Amount of time to monitor each updated task for
          failures, in nanoseconds.
        max_failure_ratio (float): The fraction of tasks that may fail during
          an update before the failure action is invoked, specified as a
          floating point number between 0 and 1. Default: 0
        order (string): Specifies the order of operations when rolling out an
          updated task. Either ``start-first`` or ``stop-first`` are accepted.
    """

    def __init__(self, parallelism=0, delay=None, failure_action='continue', monitor=None, max_failure_ratio=None, order=None):
        self['Parallelism'] = parallelism
        if delay is not None:
            self['Delay'] = delay
        if failure_action not in ('pause', 'continue', 'rollback'):
            raise errors.InvalidArgument('failure_action must be one of `pause`, `continue`, `rollback`')
        self['FailureAction'] = failure_action
        if monitor is not None:
            if not isinstance(monitor, int):
                raise TypeError('monitor must be an integer')
            self['Monitor'] = monitor
        if max_failure_ratio is not None:
            if not isinstance(max_failure_ratio, (float, int)):
                raise TypeError('max_failure_ratio must be a float')
            if max_failure_ratio > 1 or max_failure_ratio < 0:
                raise errors.InvalidArgument('max_failure_ratio must be a number between 0 and 1')
            self['MaxFailureRatio'] = max_failure_ratio
        if order is not None:
            if order not in ('start-first', 'stop-first'):
                raise errors.InvalidArgument('order must be either `start-first` or `stop-first`')
            self['Order'] = order