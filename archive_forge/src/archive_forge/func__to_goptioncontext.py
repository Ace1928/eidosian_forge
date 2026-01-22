import sys
import optparse
import warnings
from optparse import OptParseError, OptionError, OptionValueError, \
from .module import get_introspection_module
from gi import _gi, PyGIDeprecationWarning
from gi._error import GError
def _to_goptioncontext(self, values):
    if self.description:
        parameter_string = self.usage + ' - ' + self.description
    else:
        parameter_string = self.usage
    context = _gi.OptionContext(parameter_string)
    context.set_help_enabled(self.help_enabled)
    context.set_ignore_unknown_options(self.ignore_unknown_options)
    for option_group in self.option_groups:
        if isinstance(option_group, _gi.OptionGroup):
            g_group = option_group
        else:
            g_group = option_group.get_option_group(self)
        context.add_group(g_group)

    def callback(option_name, option_value, group):
        if option_name.startswith('--'):
            opt = self._long_opt[option_name]
        else:
            opt = self._short_opt[option_name]
        opt.process(option_name, option_value, values, self)
    main_group = _gi.OptionGroup(None, None, None, callback)
    main_entries = []
    for option in self.option_list:
        main_entries.extend(option._to_goptionentries())
    main_group.add_entries(main_entries)
    context.set_main_group(main_group)
    return context