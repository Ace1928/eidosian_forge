import sys, string, re
import getopt
from distutils.errors import *
def _grok_option_table(self):
    """Populate the various data structures that keep tabs on the
        option table.  Called by 'getopt()' before it can do anything
        worthwhile.
        """
    self.long_opts = []
    self.short_opts = []
    self.short2long.clear()
    self.repeat = {}
    for option in self.option_table:
        if len(option) == 3:
            long, short, help = option
            repeat = 0
        elif len(option) == 4:
            long, short, help, repeat = option
        else:
            raise ValueError('invalid option tuple: %r' % (option,))
        if not isinstance(long, str) or len(long) < 2:
            raise DistutilsGetoptError("invalid long option '%s': must be a string of length >= 2" % long)
        if not (short is None or (isinstance(short, str) and len(short) == 1)):
            raise DistutilsGetoptError("invalid short option '%s': must a single character or None" % short)
        self.repeat[long] = repeat
        self.long_opts.append(long)
        if long[-1] == '=':
            if short:
                short = short + ':'
            long = long[0:-1]
            self.takes_arg[long] = 1
        else:
            alias_to = self.negative_alias.get(long)
            if alias_to is not None:
                if self.takes_arg[alias_to]:
                    raise DistutilsGetoptError("invalid negative alias '%s': aliased option '%s' takes a value" % (long, alias_to))
                self.long_opts[-1] = long
            self.takes_arg[long] = 0
        alias_to = self.alias.get(long)
        if alias_to is not None:
            if self.takes_arg[long] != self.takes_arg[alias_to]:
                raise DistutilsGetoptError("invalid alias '%s': inconsistent with aliased option '%s' (one of them takes a value, the other doesn't" % (long, alias_to))
        if not longopt_re.match(long):
            raise DistutilsGetoptError("invalid long option name '%s' (must be letters, numbers, hyphens only" % long)
        self.attr_name[long] = self.get_attr_name(long)
        if short:
            self.short_opts.append(short)
            self.short2long[short[0]] = long