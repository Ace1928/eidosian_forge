import gc
from ..Qt import QtCore

    Disable automatic garbage collection and instead collect manually
    on a timer.

    This is done to ensure that garbage collection only happens in the GUI
    thread, as otherwise Qt can crash.

    Credit:  Erik Janssens
    Source:  http://pydev.blogspot.com/2014/03/should-python-garbage-collector-be.html
    