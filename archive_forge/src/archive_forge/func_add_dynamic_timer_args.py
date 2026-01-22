import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def add_dynamic_timer_args(self, callback, args=None, kwargs=None, initial_delay=None, periodic_interval_max=None, stop_on_exception=True):
    """Add a timer that controls its own period dynamically.

        The period of each iteration of the timer is controlled by the return
        value of the callback function on the previous iteration.

        :param callback: The callback function to run when the timer is
                         triggered.
        :param args: A list of positional args to the callback function.
        :param kwargs: A dict of keyword args to the callback function.
        :param initial_delay: The delay in seconds before first triggering the
                              timer. If not set, the timer is liable to be
                              scheduled immediately.
        :param periodic_interval_max: The maximum interval in seconds to allow
                                      the callback function to request. If
                                      provided, this is also used as the
                                      default delay if None is returned by the
                                      callback function.
        :param stop_on_exception: Pass ``False`` to have the timer continue
                                  running even if the callback function raises
                                  an exception.
        :returns: an :class:`oslo_service.loopingcall.DynamicLoopingCall`
                  instance
        """
    args = args or []
    kwargs = kwargs or {}
    timer = loopingcall.DynamicLoopingCall(callback, *args, **kwargs)
    timer.start(initial_delay=initial_delay, periodic_interval_max=periodic_interval_max, stop_on_exception=stop_on_exception)
    self.timers.append(timer)
    return timer