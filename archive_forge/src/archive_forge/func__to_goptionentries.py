import sys
import optparse
import warnings
from optparse import OptParseError, OptionError, OptionValueError, \
from .module import get_introspection_module
from gi import _gi, PyGIDeprecationWarning
from gi._error import GError
def _to_goptionentries(self):
    flags = 0
    if self.hidden:
        flags |= GLib.OptionFlags.HIDDEN
    if self.in_main:
        flags |= GLib.OptionFlags.IN_MAIN
    if self.takes_value():
        if self.optional_arg:
            flags |= GLib.OptionFlags.OPTIONAL_ARG
    else:
        flags |= GLib.OptionFlags.NO_ARG
    if self.type == 'filename':
        flags |= GLib.OptionFlags.FILENAME
    for long_name, short_name in zip(self._long_opts, self._short_opts):
        short_bytes = short_name[1]
        if not isinstance(short_bytes, bytes):
            short_bytes = short_bytes.encode('utf-8')
        yield (long_name[2:], short_bytes, flags, self.help, self.metavar)
    for long_name in self._long_opts[len(self._short_opts):]:
        yield (long_name[2:], b'\x00', flags, self.help, self.metavar)