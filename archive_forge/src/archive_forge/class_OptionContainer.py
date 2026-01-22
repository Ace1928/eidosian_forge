import sys, os
import textwrap
class OptionContainer:
    """
    Abstract base class.

    Class attributes:
      standard_option_list : [Option]
        list of standard options that will be accepted by all instances
        of this parser class (intended to be overridden by subclasses).

    Instance attributes:
      option_list : [Option]
        the list of Option objects contained by this OptionContainer
      _short_opt : { string : Option }
        dictionary mapping short option strings, eg. "-f" or "-X",
        to the Option instances that implement them.  If an Option
        has multiple short option strings, it will appear in this
        dictionary multiple times. [1]
      _long_opt : { string : Option }
        dictionary mapping long option strings, eg. "--file" or
        "--exclude", to the Option instances that implement them.
        Again, a given Option can occur multiple times in this
        dictionary. [1]
      defaults : { string : any }
        dictionary mapping option destination names to default
        values for each destination [1]

    [1] These mappings are common to (shared by) all components of the
        controlling OptionParser, where they are initially created.

    """

    def __init__(self, option_class, conflict_handler, description):
        self._create_option_list()
        self.option_class = option_class
        self.set_conflict_handler(conflict_handler)
        self.set_description(description)

    def _create_option_mappings(self):
        self._short_opt = {}
        self._long_opt = {}
        self.defaults = {}

    def _share_option_mappings(self, parser):
        self._short_opt = parser._short_opt
        self._long_opt = parser._long_opt
        self.defaults = parser.defaults

    def set_conflict_handler(self, handler):
        if handler not in ('error', 'resolve'):
            raise ValueError('invalid conflict_resolution value %r' % handler)
        self.conflict_handler = handler

    def set_description(self, description):
        self.description = description

    def get_description(self):
        return self.description

    def destroy(self):
        """see OptionParser.destroy()."""
        del self._short_opt
        del self._long_opt
        del self.defaults

    def _check_conflict(self, option):
        conflict_opts = []
        for opt in option._short_opts:
            if opt in self._short_opt:
                conflict_opts.append((opt, self._short_opt[opt]))
        for opt in option._long_opts:
            if opt in self._long_opt:
                conflict_opts.append((opt, self._long_opt[opt]))
        if conflict_opts:
            handler = self.conflict_handler
            if handler == 'error':
                raise OptionConflictError('conflicting option string(s): %s' % ', '.join([co[0] for co in conflict_opts]), option)
            elif handler == 'resolve':
                for opt, c_option in conflict_opts:
                    if opt.startswith('--'):
                        c_option._long_opts.remove(opt)
                        del self._long_opt[opt]
                    else:
                        c_option._short_opts.remove(opt)
                        del self._short_opt[opt]
                    if not (c_option._short_opts or c_option._long_opts):
                        c_option.container.option_list.remove(c_option)

    def add_option(self, *args, **kwargs):
        """add_option(Option)
           add_option(opt_str, ..., kwarg=val, ...)
        """
        if isinstance(args[0], str):
            option = self.option_class(*args, **kwargs)
        elif len(args) == 1 and (not kwargs):
            option = args[0]
            if not isinstance(option, Option):
                raise TypeError('not an Option instance: %r' % option)
        else:
            raise TypeError('invalid arguments')
        self._check_conflict(option)
        self.option_list.append(option)
        option.container = self
        for opt in option._short_opts:
            self._short_opt[opt] = option
        for opt in option._long_opts:
            self._long_opt[opt] = option
        if option.dest is not None:
            if option.default is not NO_DEFAULT:
                self.defaults[option.dest] = option.default
            elif option.dest not in self.defaults:
                self.defaults[option.dest] = None
        return option

    def add_options(self, option_list):
        for option in option_list:
            self.add_option(option)

    def get_option(self, opt_str):
        return self._short_opt.get(opt_str) or self._long_opt.get(opt_str)

    def has_option(self, opt_str):
        return opt_str in self._short_opt or opt_str in self._long_opt

    def remove_option(self, opt_str):
        option = self._short_opt.get(opt_str)
        if option is None:
            option = self._long_opt.get(opt_str)
        if option is None:
            raise ValueError('no such option %r' % opt_str)
        for opt in option._short_opts:
            del self._short_opt[opt]
        for opt in option._long_opts:
            del self._long_opt[opt]
        option.container.option_list.remove(option)

    def format_option_help(self, formatter):
        if not self.option_list:
            return ''
        result = []
        for option in self.option_list:
            if not option.help is SUPPRESS_HELP:
                result.append(formatter.format_option(option))
        return ''.join(result)

    def format_description(self, formatter):
        return formatter.format_description(self.get_description())

    def format_help(self, formatter):
        result = []
        if self.description:
            result.append(self.format_description(formatter))
        if self.option_list:
            result.append(self.format_option_help(formatter))
        return '\n'.join(result)