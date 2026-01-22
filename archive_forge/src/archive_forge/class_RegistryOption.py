import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
class RegistryOption(Option):
    """Option based on a registry

    The values for the options correspond to entries in the registry.  Input
    must be a registry key.  After validation, it is converted into an object
    using Registry.get or a caller-provided converter.
    """

    def validate_value(self, value):
        """Validate a value name"""
        if value not in self.registry:
            raise BadOptionValue(self.name, value)

    def convert(self, value):
        """Convert a value name into an output type"""
        self.validate_value(value)
        if self.converter is None:
            return self.registry.get(value)
        else:
            return self.converter(value)

    def __init__(self, name, help, registry=None, converter=None, value_switches=False, title=None, enum_switch=True, lazy_registry=None, short_name=None, short_value_switches=None):
        """
        Constructor.

        Args:
          name: The option name.
          help: Help for the option.
          registry: A Registry containing the values
          converter: Callable to invoke with the value name to produce
            the value.  If not supplied, self.registry.get is used.
          value_switches: If true, each possible value is assigned its
            own switch.  For example, instead of '--format knit',
            '--knit' can be used interchangeably.
          enum_switch: If true, a switch is provided with the option name,
            which takes a value.
          lazy_registry: A tuple of (module name, attribute name) for a
            registry to be lazily loaded.
          short_name: The short name for the enum switch, if any
          short_value_switches: A dict mapping values to short names
        """
        Option.__init__(self, name, help, type=self.convert, short_name=short_name)
        self._registry = registry
        if registry is None:
            if lazy_registry is None:
                raise AssertionError('One of registry or lazy_registry must be given.')
            self._lazy_registry = _mod_registry._LazyObjectGetter(*lazy_registry)
        if registry is not None and lazy_registry is not None:
            raise AssertionError('registry and lazy_registry are mutually exclusive')
        self.name = name
        self.converter = converter
        self.value_switches = value_switches
        self.enum_switch = enum_switch
        self.short_value_switches = short_value_switches
        self.title = title
        if self.title is None:
            self.title = name

    @property
    def registry(self):
        if self._registry is None:
            self._registry = self._lazy_registry.get_obj()
        return self._registry

    @staticmethod
    def from_kwargs(name_, help=None, title=None, value_switches=False, enum_switch=True, **kwargs):
        """Convenience method to generate string-map registry options

        name, help, value_switches and enum_switch are passed to the
        RegistryOption constructor.  Any other keyword arguments are treated
        as values for the option, and their value is treated as the help.
        """
        reg = _mod_registry.Registry()
        for name, switch_help in sorted(kwargs.items()):
            name = name.replace('_', '-')
            reg.register(name, name, help=switch_help)
            if not value_switches:
                help = help + '  "' + name + '": ' + switch_help
                if not help.endswith('.'):
                    help = help + '.'
        return RegistryOption(name_, help, reg, title=title, value_switches=value_switches, enum_switch=enum_switch)

    def add_option(self, parser, short_name):
        """Add this option to an Optparse parser"""
        if self.value_switches:
            parser = parser.add_option_group(self.title)
        if self.enum_switch:
            Option.add_option(self, parser, short_name)
        if self.value_switches:
            alias_map = self.registry.alias_map()
            for key in self.registry.keys():
                if key in self.registry.aliases():
                    continue
                option_strings = ['--%s' % name for name in [key] + [alias for alias in alias_map.get(key, []) if not self.is_hidden(alias)]]
                if self.is_hidden(key):
                    help = optparse.SUPPRESS_HELP
                else:
                    help = self.registry.get_help(key)
                if self.short_value_switches and key in self.short_value_switches:
                    option_strings.append('-%s' % self.short_value_switches[key])
                parser.add_option(*option_strings, action='callback', callback=self._optparse_value_callback(key), help=help)

    def _optparse_value_callback(self, cb_value):

        def cb(option, opt, value, parser):
            v = self.type(cb_value)
            setattr(parser.values, self._param_name, v)
            if self.custom_callback is not None:
                self.custom_callback(option, self._param_name, v, parser)
        return cb

    def iter_switches(self):
        """Iterate through the list of switches provided by the option

        :return: an iterator of (name, short_name, argname, help)
        """
        yield from Option.iter_switches(self)
        if self.value_switches:
            for key in sorted(self.registry.keys()):
                yield (key, None, None, self.registry.get_help(key))

    def is_alias(self, name):
        """Check whether a particular format is an alias."""
        if name == self.name:
            return False
        return name in self.registry.aliases()

    def is_hidden(self, name):
        if name == self.name:
            return Option.is_hidden(self, name)
        return getattr(self.registry.get_info(name), 'hidden', False)