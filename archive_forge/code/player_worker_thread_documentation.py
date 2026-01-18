import threading
import time
from typing import TYPE_CHECKING, Set
import pyglet
from pyglet.util import debug_print

        Remove a player from the PlayerWorkerThread, or ignore if it does
        not exist.

        Do not call this method from within the thread, as it may deadlock.
        