from __future__ import generators
from dbus.exceptions import DBusException
from _dbus_bindings import (
from dbus.bus import BusConnection
from dbus.lowlevel import SignalMessage
from dbus._compat import is_py2
class StarterBus(Bus):
    """The bus that activated this process (only valid if
    this process was launched by DBus activation).
    """

    def __new__(cls, private=False, mainloop=None):
        """Return a connection to the bus that activated this process.

        :Parameters:
            `private` : bool
                If true, never return an existing shared instance, but instead
                return a private connection.
            `mainloop` : dbus.mainloop.NativeMainLoop
                The main loop to use. The default is to use the default
                main loop if one has been set up, or raise an exception
                if none has been.
        """
        return Bus.__new__(cls, Bus.TYPE_STARTER, private=private, mainloop=mainloop)