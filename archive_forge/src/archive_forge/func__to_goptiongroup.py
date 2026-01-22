import sys
import optparse
import warnings
from optparse import OptParseError, OptionError, OptionValueError, \
from .module import get_introspection_module
from gi import _gi, PyGIDeprecationWarning
from gi._error import GError
def _to_goptiongroup(self, parser):

    def callback(option_name, option_value, group):
        if option_name.startswith('--'):
            opt = self._long_opt[option_name]
        else:
            opt = self._short_opt[option_name]
        try:
            opt.process(option_name, option_value, self.values, parser)
        except OptionValueError:
            error = sys.exc_info()[1]
            gerror = GError(str(error))
            gerror.domain = OPTION_CONTEXT_ERROR_QUARK
            gerror.code = GLib.OptionError.BAD_VALUE
            gerror.message = str(error)
            raise gerror
    group = _gi.OptionGroup(self.name, self.description, self.help_description, callback)
    if self.translation_domain:
        group.set_translation_domain(self.translation_domain)
    entries = []
    for option in self.option_list:
        entries.extend(option._to_goptionentries())
    group.add_entries(entries)
    return group