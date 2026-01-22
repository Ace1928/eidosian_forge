from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
class ArgParseConfigLoader(CommandLineConfigLoader):
    """A loader that uses the argparse module to load from the command line."""
    parser_class = ArgumentParser

    def __init__(self, argv: list[str] | None=None, aliases: dict[str, str] | None=None, flags: dict[str, str] | None=None, log: t.Any=None, classes: list[type[t.Any]] | None=None, subcommands: SubcommandsDict | None=None, *parser_args: t.Any, **parser_kw: t.Any) -> None:
        """Create a config loader for use with argparse.

        Parameters
        ----------
        classes : optional, list
            The classes to scan for *container* config-traits and decide
            for their "multiplicity" when adding them as *argparse* arguments.
        argv : optional, list
            If given, used to read command-line arguments from, otherwise
            sys.argv[1:] is used.
        *parser_args : tuple
            A tuple of positional arguments that will be passed to the
            constructor of :class:`argparse.ArgumentParser`.
        **parser_kw : dict
            A tuple of keyword arguments that will be passed to the
            constructor of :class:`argparse.ArgumentParser`.
        aliases : dict of str to str
            Dict of aliases to full traitlets names for CLI parsing
        flags : dict of str to str
            Dict of flags to full traitlets names for CLI parsing
        log
            Passed to `ConfigLoader`

        Returns
        -------
        config : Config
            The resulting Config object.
        """
        classes = classes or []
        super(CommandLineConfigLoader, self).__init__(log=log)
        self.clear()
        if argv is None:
            argv = sys.argv[1:]
        self.argv = argv
        self.aliases = aliases or {}
        self.flags = flags or {}
        self.classes = classes
        self.subcommands = subcommands
        self.parser_args = parser_args
        self.version = parser_kw.pop('version', None)
        kwargs = dict(argument_default=argparse.SUPPRESS)
        kwargs.update(parser_kw)
        self.parser_kw = kwargs

    def load_config(self, argv: list[str] | None=None, aliases: t.Any=None, flags: t.Any=_deprecated, classes: t.Any=None) -> Config:
        """Parse command line arguments and return as a Config object.

        Parameters
        ----------
        argv : optional, list
            If given, a list with the structure of sys.argv[1:] to parse
            arguments from. If not given, the instance's self.argv attribute
            (given at construction time) is used.
        flags
            Deprecated in traitlets 5.0, instantiate the config loader with the flags.

        """
        if flags is not _deprecated:
            warnings.warn(f'The `flag` argument to load_config is deprecated since Traitlets 5.0 and will be ignored, pass flags the `{type(self)}` constructor.', DeprecationWarning, stacklevel=2)
        self.clear()
        if argv is None:
            argv = self.argv
        if aliases is not None:
            self.aliases = aliases
        if classes is not None:
            self.classes = classes
        self._create_parser()
        self._argcomplete(self.classes, self.subcommands)
        self._parse_args(argv)
        self._convert_to_config()
        return self.config

    def get_extra_args(self) -> list[str]:
        if hasattr(self, 'extra_args'):
            return self.extra_args
        else:
            return []

    def _create_parser(self) -> None:
        self.parser = self.parser_class(*self.parser_args, **self.parser_kw)
        self._add_arguments(self.aliases, self.flags, self.classes)

    def _add_arguments(self, aliases: t.Any, flags: t.Any, classes: t.Any) -> None:
        raise NotImplementedError('subclasses must implement _add_arguments')

    def _argcomplete(self, classes: list[t.Any], subcommands: SubcommandsDict | None) -> None:
        """If argcomplete is enabled, allow triggering command-line autocompletion"""

    def _parse_args(self, args: t.Any) -> t.Any:
        """self.parser->self.parsed_data"""
        uargs = [cast_unicode(a) for a in args]
        unpacked_aliases: dict[str, str] = {}
        if self.aliases:
            unpacked_aliases = {}
            for alias, alias_target in self.aliases.items():
                if alias in self.flags:
                    continue
                if not isinstance(alias, tuple):
                    alias = (alias,)
                for al in alias:
                    if len(al) == 1:
                        unpacked_aliases['-' + al] = '--' + alias_target
                    unpacked_aliases['--' + al] = '--' + alias_target

        def _replace(arg: str) -> str:
            if arg == '-':
                return _DASH_REPLACEMENT
            for k, v in unpacked_aliases.items():
                if arg == k:
                    return v
                if arg.startswith(k + '='):
                    return v + '=' + arg[len(k) + 1:]
            return arg
        if '--' in uargs:
            idx = uargs.index('--')
            extra_args = uargs[idx + 1:]
            to_parse = uargs[:idx]
        else:
            extra_args = []
            to_parse = uargs
        to_parse = [_replace(a) for a in to_parse]
        self.parsed_data = self.parser.parse_args(to_parse)
        self.extra_args = extra_args

    def _convert_to_config(self) -> None:
        """self.parsed_data->self.config"""
        for k, v in vars(self.parsed_data).items():
            *path, key = k.split('.')
            section = self.config
            for p in path:
                section = section[p]
            setattr(section, key, v)