import logging
import os
import pathlib
import site
import sys
import textwrap
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Type, Union
from pip._vendor.certifi import where
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version
from pip import __file__ as pip_location
from pip._internal.cli.spinners import open_spinner
from pip._internal.locations import get_platlib, get_purelib, get_scheme
from pip._internal.metadata import get_default_environment, get_environment
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
@staticmethod
def _install_requirements(pip_runnable: str, finder: 'PackageFinder', requirements: Iterable[str], prefix: _Prefix, *, kind: str) -> None:
    args: List[str] = [sys.executable, pip_runnable, 'install', '--ignore-installed', '--no-user', '--prefix', prefix.path, '--no-warn-script-location']
    if logger.getEffectiveLevel() <= logging.DEBUG:
        args.append('-v')
    for format_control in ('no_binary', 'only_binary'):
        formats = getattr(finder.format_control, format_control)
        args.extend(('--' + format_control.replace('_', '-'), ','.join(sorted(formats or {':none:'}))))
    index_urls = finder.index_urls
    if index_urls:
        args.extend(['-i', index_urls[0]])
        for extra_index in index_urls[1:]:
            args.extend(['--extra-index-url', extra_index])
    else:
        args.append('--no-index')
    for link in finder.find_links:
        args.extend(['--find-links', link])
    for host in finder.trusted_hosts:
        args.extend(['--trusted-host', host])
    if finder.allow_all_prereleases:
        args.append('--pre')
    if finder.prefer_binary:
        args.append('--prefer-binary')
    args.append('--')
    args.extend(requirements)
    extra_environ = {'_PIP_STANDALONE_CERT': where()}
    with open_spinner(f'Installing {kind}') as spinner:
        call_subprocess(args, command_desc=f'pip subprocess to install {kind}', spinner=spinner, extra_environ=extra_environ)