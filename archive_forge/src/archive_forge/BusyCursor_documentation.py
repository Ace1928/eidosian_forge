from contextlib import contextmanager
from ..Qt import QtCore, QtGui, QtWidgets

    Display a busy mouse cursor during long operations.
    Usage::

        with BusyCursor():
            doLongOperation()

    May be nested. If called from a non-gui thread, then the cursor will not be affected.
    