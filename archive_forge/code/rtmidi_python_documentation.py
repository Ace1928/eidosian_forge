import queue
import rtmidi_python as rtmidi
from ..ports import BaseInput, BaseOutput
Backend for rtmidi-python:

https://pypi.python.org/pypi/rtmidi-python

To use this backend copy (or link) it to somewhere in your Python path
and call:

    mido.set_backend('mido.backends.rtmidi_python')

or set shell variable $MIDO_BACKEND to mido.backends.rtmidi_python

TODO:

* add support for APIs.

* active_sensing is still filtered. (The same is true for
  mido.backends.rtmidi.)There may be a way to remove this filtering.

