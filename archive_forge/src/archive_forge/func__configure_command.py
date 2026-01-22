from __future__ import annotations
import datetime
import fnmatch
import logging
import optparse
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from configparser import RawConfigParser
from io import StringIO
from typing import Iterable
from babel import Locale, localedata
from babel import __version__ as VERSION
from babel.core import UnknownLocaleError
from babel.messages.catalog import DEFAULT_HEADER, Catalog
from babel.messages.extract import (
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po, write_po
from babel.util import LOCALTZ
def _configure_command(self, cmdname, argv):
    """
        :type cmdname: str
        :type argv: list[str]
        """
    cmdclass = self.command_classes[cmdname]
    cmdinst = cmdclass()
    if self.log:
        cmdinst.log = self.log
    assert isinstance(cmdinst, CommandMixin)
    cmdinst.initialize_options()
    parser = optparse.OptionParser(usage=self.usage % (cmdname, ''), description=self.commands[cmdname])
    as_args = getattr(cmdclass, 'as_args', ())
    for long, short, help in cmdclass.user_options:
        name = long.strip('=')
        default = getattr(cmdinst, name.replace('-', '_'))
        strs = [f'--{name}']
        if short:
            strs.append(f'-{short}')
        strs.extend(cmdclass.option_aliases.get(name, ()))
        choices = cmdclass.option_choices.get(name, None)
        if name == as_args:
            parser.usage += f'<{name}>'
        elif name in cmdclass.boolean_options:
            parser.add_option(*strs, action='store_true', help=help)
        elif name in cmdclass.multiple_value_options:
            parser.add_option(*strs, action='append', help=help, choices=choices)
        else:
            parser.add_option(*strs, help=help, default=default, choices=choices)
    options, args = parser.parse_args(argv)
    if as_args:
        setattr(options, as_args.replace('-', '_'), args)
    for key, value in vars(options).items():
        setattr(cmdinst, key, value)
    try:
        cmdinst.ensure_finalized()
    except OptionError as err:
        parser.error(str(err))
    return cmdinst