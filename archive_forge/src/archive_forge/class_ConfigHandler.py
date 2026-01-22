import contextlib
import functools
import os
from collections import defaultdict
from functools import partial
from functools import wraps
from typing import (
from ..errors import FileError, OptionError
from ..extern.packaging.markers import default_environment as marker_env
from ..extern.packaging.requirements import InvalidRequirement, Requirement
from ..extern.packaging.specifiers import SpecifierSet
from ..extern.packaging.version import InvalidVersion, Version
from ..warnings import SetuptoolsDeprecationWarning
from . import expand
class ConfigHandler(Generic[Target]):
    """Handles metadata supplied in configuration files."""
    section_prefix: str
    'Prefix for config sections handled by this handler.\n    Must be provided by class heirs.\n\n    '
    aliases: Dict[str, str] = {}
    'Options aliases.\n    For compatibility with various packages. E.g.: d2to1 and pbr.\n    Note: `-` in keys is replaced with `_` by config parser.\n\n    '

    def __init__(self, target_obj: Target, options: AllCommandOptions, ignore_option_errors, ensure_discovered: expand.EnsurePackagesDiscovered):
        self.ignore_option_errors = ignore_option_errors
        self.target_obj = target_obj
        self.sections = dict(self._section_options(options))
        self.set_options: List[str] = []
        self.ensure_discovered = ensure_discovered
        self._referenced_files: Set[str] = set()
        'After parsing configurations, this property will enumerate\n        all files referenced by the "file:" directive. Private API for setuptools only.\n        '

    @classmethod
    def _section_options(cls, options: AllCommandOptions):
        for full_name, value in options.items():
            pre, sep, name = full_name.partition(cls.section_prefix)
            if pre:
                continue
            yield (name.lstrip('.'), value)

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        raise NotImplementedError('%s must provide .parsers property' % self.__class__.__name__)

    def __setitem__(self, option_name, value):
        target_obj = self.target_obj
        option_name = self.aliases.get(option_name, option_name)
        try:
            current_value = getattr(target_obj, option_name)
        except AttributeError as e:
            raise KeyError(option_name) from e
        if current_value:
            return
        try:
            parsed = self.parsers.get(option_name, lambda x: x)(value)
        except (Exception,) * self.ignore_option_errors:
            return
        simple_setter = functools.partial(target_obj.__setattr__, option_name)
        setter = getattr(target_obj, 'set_%s' % option_name, simple_setter)
        setter(parsed)
        self.set_options.append(option_name)

    @classmethod
    def _parse_list(cls, value, separator=','):
        """Represents value as a list.

        Value is split either by separator (defaults to comma) or by lines.

        :param value:
        :param separator: List items separator character.
        :rtype: list
        """
        if isinstance(value, list):
            return value
        if '\n' in value:
            value = value.splitlines()
        else:
            value = value.split(separator)
        return [chunk.strip() for chunk in value if chunk.strip()]

    @classmethod
    def _parse_dict(cls, value):
        """Represents value as a dict.

        :param value:
        :rtype: dict
        """
        separator = '='
        result = {}
        for line in cls._parse_list(value):
            key, sep, val = line.partition(separator)
            if sep != separator:
                raise OptionError(f'Unable to parse option value to dict: {value}')
            result[key.strip()] = val.strip()
        return result

    @classmethod
    def _parse_bool(cls, value):
        """Represents value as boolean.

        :param value:
        :rtype: bool
        """
        value = value.lower()
        return value in ('1', 'true', 'yes')

    @classmethod
    def _exclude_files_parser(cls, key):
        """Returns a parser function to make sure field inputs
        are not files.

        Parses a value after getting the key so error messages are
        more informative.

        :param key:
        :rtype: callable
        """

        def parser(value):
            exclude_directive = 'file:'
            if value.startswith(exclude_directive):
                raise ValueError('Only strings are accepted for the {0} field, files are not accepted'.format(key))
            return value
        return parser

    def _parse_file(self, value, root_dir: _Path):
        """Represents value as a string, allowing including text
        from nearest files using `file:` directive.

        Directive is sandboxed and won't reach anything outside
        directory with setup.py.

        Examples:
            file: README.rst, CHANGELOG.md, src/file.txt

        :param str value:
        :rtype: str
        """
        include_directive = 'file:'
        if not isinstance(value, str):
            return value
        if not value.startswith(include_directive):
            return value
        spec = value[len(include_directive):]
        filepaths = [path.strip() for path in spec.split(',')]
        self._referenced_files.update(filepaths)
        return expand.read_files(filepaths, root_dir)

    def _parse_attr(self, value, package_dir, root_dir: _Path):
        """Represents value as a module attribute.

        Examples:
            attr: package.attr
            attr: package.module.attr

        :param str value:
        :rtype: str
        """
        attr_directive = 'attr:'
        if not value.startswith(attr_directive):
            return value
        attr_desc = value.replace(attr_directive, '')
        package_dir.update(self.ensure_discovered.package_dir)
        return expand.read_attr(attr_desc, package_dir, root_dir)

    @classmethod
    def _get_parser_compound(cls, *parse_methods):
        """Returns parser function to represents value as a list.

        Parses a value applying given methods one after another.

        :param parse_methods:
        :rtype: callable
        """

        def parse(value):
            parsed = value
            for method in parse_methods:
                parsed = method(parsed)
            return parsed
        return parse

    @classmethod
    def _parse_section_to_dict_with_key(cls, section_options, values_parser):
        """Parses section options into a dictionary.

        Applies a given parser to each option in a section.

        :param dict section_options:
        :param callable values_parser: function with 2 args corresponding to key, value
        :rtype: dict
        """
        value = {}
        for key, (_, val) in section_options.items():
            value[key] = values_parser(key, val)
        return value

    @classmethod
    def _parse_section_to_dict(cls, section_options, values_parser=None):
        """Parses section options into a dictionary.

        Optionally applies a given parser to each value.

        :param dict section_options:
        :param callable values_parser: function with 1 arg corresponding to option value
        :rtype: dict
        """
        parser = (lambda _, v: values_parser(v)) if values_parser else lambda _, v: v
        return cls._parse_section_to_dict_with_key(section_options, parser)

    def parse_section(self, section_options):
        """Parses configuration file section.

        :param dict section_options:
        """
        for name, (_, value) in section_options.items():
            with contextlib.suppress(KeyError):
                self[name] = value

    def parse(self):
        """Parses configuration file items from one
        or more related sections.

        """
        for section_name, section_options in self.sections.items():
            method_postfix = ''
            if section_name:
                method_postfix = '_%s' % section_name
            section_parser_method: Optional[Callable] = getattr(self, ('parse_section%s' % method_postfix).replace('.', '__'), None)
            if section_parser_method is None:
                raise OptionError(f'Unsupported distribution option section: [{self.section_prefix}.{section_name}]')
            section_parser_method(section_options)

    def _deprecated_config_handler(self, func, msg, **kw):
        """this function will wrap around parameters that are deprecated

        :param msg: deprecation message
        :param func: function to be wrapped around
        """

        @wraps(func)
        def config_handler(*args, **kwargs):
            kw.setdefault('stacklevel', 2)
            _DeprecatedConfig.emit('Deprecated config in `setup.cfg`', msg, **kw)
            return func(*args, **kwargs)
        return config_handler