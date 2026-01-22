import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def add_timer_args(self, interval, callback, args=None, kwargs=None, initial_delay=None, stop_on_exception=True):
    """Add a timer with a fixed period.

        :param interval: The minimum period in seconds between calls to the
                         callback function.
        :param callback: The callback function to run when the timer is
                         triggered.
        :param args: A list of positional args to the callback function.
        :param kwargs: A dict of keyword args to the callback function.
        :param initial_delay: The delay in seconds before first triggering the
                              timer. If not set, the timer is liable to be
                              scheduled immediately.
        :param stop_on_exception: Pass ``False`` to have the timer continue
                                  running even if the callback function raises
                                  an exception.
        :returns: an :class:`oslo_service.loopingcall.FixedIntervalLoopingCall`
                  instance
        """
    args = args or []
    kwargs = kwargs or {}
    pulse = loopingcall.FixedIntervalLoopingCall(callback, *args, **kwargs)
    pulse.start(interval=interval, initial_delay=initial_delay, stop_on_exception=stop_on_exception)
    self.timers.append(pulse)
    return pulse