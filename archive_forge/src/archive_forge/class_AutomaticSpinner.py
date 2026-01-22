import multiprocessing
import sys
import time
from humanfriendly import Timer
from humanfriendly.deprecation import deprecated_args
from humanfriendly.terminal import ANSI_ERASE_LINE
class AutomaticSpinner(object):
    """
    Show a spinner on the terminal that automatically starts animating.

    This class shows a spinner on the terminal (just like :class:`Spinner`
    does) that automatically starts animating. This class should be used as a
    context manager using the :keyword:`with` statement. The animation
    continues for as long as the context is active.

    :class:`AutomaticSpinner` provides an alternative to :class:`Spinner`
    for situations where it is not practical for the caller to periodically
    call :func:`~Spinner.step()` to advance the animation, e.g. because
    you're performing a blocking call and don't fancy implementing threading or
    subprocess handling just to provide some user feedback.

    This works using the :mod:`multiprocessing` module by spawning a
    subprocess to render the spinner while the main process is busy doing
    something more useful. By using the :keyword:`with` statement you're
    guaranteed that the subprocess is properly terminated at the appropriate
    time.
    """

    def __init__(self, label, show_time=True):
        """
        Initialize an automatic spinner.

        :param label: The label for the spinner (a string).
        :param show_time: If this is :data:`True` (the default) then the spinner
                          shows elapsed time.
        """
        self.label = label
        self.show_time = show_time
        self.shutdown_event = multiprocessing.Event()
        self.subprocess = multiprocessing.Process(target=self._target)

    def __enter__(self):
        """Enable the use of automatic spinners as context managers."""
        self.subprocess.start()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Enable the use of automatic spinners as context managers."""
        self.shutdown_event.set()
        self.subprocess.join()

    def _target(self):
        try:
            timer = Timer() if self.show_time else None
            with Spinner(label=self.label, timer=timer) as spinner:
                while not self.shutdown_event.is_set():
                    spinner.step()
                    spinner.sleep()
        except KeyboardInterrupt:
            pass