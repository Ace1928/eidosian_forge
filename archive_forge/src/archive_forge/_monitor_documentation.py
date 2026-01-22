import atexit
from threading import Event, Thread, current_thread
from time import time
from warnings import warn

    Monitoring thread for tqdm bars.
    Monitors if tqdm bars are taking too much time to display
    and readjusts miniters automatically if necessary.

    Parameters
    ----------
    tqdm_cls  : class
        tqdm class to use (can be core tqdm or a submodule).
    sleep_interval  : float
        Time to sleep between monitoring checks.
    