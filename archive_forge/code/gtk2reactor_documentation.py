from incremental import Version
from ._deprecate import deprecatedGnomeReactor
import sys
from twisted.internet import _glibbase
from twisted.python import runtime
import gobject

    Configure the twisted mainloop to be run inside the gtk mainloop.
    