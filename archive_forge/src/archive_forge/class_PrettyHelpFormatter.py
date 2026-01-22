import logging
import optparse
import shutil
import sys
import textwrap
from contextlib import suppress
from typing import Any, Dict, Generator, List, Tuple
from pip._internal.cli.status_codes import UNKNOWN_ERROR
from pip._internal.configuration import Configuration, ConfigurationError
from pip._internal.utils.misc import redact_auth_from_url, strtobool
class PrettyHelpFormatter(optparse.IndentedHelpFormatter):
    """A prettier/less verbose help formatter for optparse."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['max_help_position'] = 30
        kwargs['indent_increment'] = 1
        kwargs['width'] = shutil.get_terminal_size()[0] - 2
        super().__init__(*args, **kwargs)

    def format_option_strings(self, option: optparse.Option) -> str:
        return self._format_option_strings(option)

    def _format_option_strings(self, option: optparse.Option, mvarfmt: str=' <{}>', optsep: str=', ') -> str:
        """
        Return a comma-separated list of option strings and metavars.

        :param option:  tuple of (short opt, long opt), e.g: ('-f', '--format')
        :param mvarfmt: metavar format string
        :param optsep:  separator
        """
        opts = []
        if option._short_opts:
            opts.append(option._short_opts[0])
        if option._long_opts:
            opts.append(option._long_opts[0])
        if len(opts) > 1:
            opts.insert(1, optsep)
        if option.takes_value():
            assert option.dest is not None
            metavar = option.metavar or option.dest.lower()
            opts.append(mvarfmt.format(metavar.lower()))
        return ''.join(opts)

    def format_heading(self, heading: str) -> str:
        if heading == 'Options':
            return ''
        return heading + ':\n'

    def format_usage(self, usage: str) -> str:
        """
        Ensure there is only one newline between usage and the first heading
        if there is no description.
        """
        msg = '\nUsage: {}\n'.format(self.indent_lines(textwrap.dedent(usage), '  '))
        return msg

    def format_description(self, description: str) -> str:
        if description:
            if hasattr(self.parser, 'main'):
                label = 'Commands'
            else:
                label = 'Description'
            description = description.lstrip('\n')
            description = description.rstrip()
            description = self.indent_lines(textwrap.dedent(description), '  ')
            description = f'{label}:\n{description}\n'
            return description
        else:
            return ''

    def format_epilog(self, epilog: str) -> str:
        if epilog:
            return epilog
        else:
            return ''

    def indent_lines(self, text: str, indent: str) -> str:
        new_lines = [indent + line for line in text.split('\n')]
        return '\n'.join(new_lines)