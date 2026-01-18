import logging
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.delayed_queue import DelayedQueue
from wandb_watchdog.observers.inotify_c import Inotify
Read event from `inotify` and add them to `queue`. When reading a
        IN_MOVE_TO event, remove the previous added matching IN_MOVE_FROM event
        and add them back to the queue as a tuple.
        