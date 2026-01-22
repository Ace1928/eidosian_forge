import codecs
import os
import pipes
import re
import subprocess
import tempfile
from humanfriendly.terminal import (
from coloredlogs.converter.colors import (
class ColoredCronMailer(object):
    """
    Easy to use integration between :mod:`coloredlogs` and the UNIX ``cron`` daemon.

    By using :class:`ColoredCronMailer` as a context manager in the command
    line interface of your Python program you make it trivially easy for users
    of your program to opt in to HTML output under ``cron``: The only thing the
    user needs to do is set ``CONTENT_TYPE="text/html"`` in their crontab!

    Under the hood this requires quite a bit of magic and I must admit that I
    developed this code simply because I was curious whether it could even be
    done :-). It requires my :mod:`capturer` package which you can install
    using ``pip install 'coloredlogs[cron]'``. The ``[cron]`` extra will pull
    in the :mod:`capturer` 2.4 or newer which is required to capture the output
    while silencing it - otherwise you'd get duplicate output in the emails
    sent by ``cron``.
    """

    def __init__(self):
        """Initialize output capturing when running under ``cron`` with the correct configuration."""
        self.is_enabled = 'text/html' in os.environ.get('CONTENT_TYPE', 'text/plain')
        self.is_silent = False
        if self.is_enabled:
            from capturer import CaptureOutput
            self.capturer = CaptureOutput(merged=True, relay=False)

    def __enter__(self):
        """Start capturing output (when applicable)."""
        if self.is_enabled:
            self.capturer.__enter__()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Stop capturing output and convert the output to HTML (when applicable)."""
        if self.is_enabled:
            if not self.is_silent:
                text = self.capturer.get_text()
                if text and (not text.isspace()):
                    output(convert(text))
            self.capturer.__exit__(exc_type, exc_value, traceback)

    def silence(self):
        """
        Tell :func:`__exit__()` to swallow all output (things will be silent).

        This can be useful when a Python program is written in such a way that
        it has already produced output by the time it becomes apparent that
        nothing useful can be done (say in a cron job that runs every few
        minutes :-p). By calling :func:`silence()` the output can be swallowed
        retroactively, avoiding useless emails from ``cron``.
        """
        self.is_silent = True