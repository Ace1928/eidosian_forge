import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
class LanguageServerManagerAPI(LoggingConfigurable, HasListeners):
    """Public API that can be used for python-based spec finders and listeners"""
    language_servers: KeyedLanguageServerSpecs
    nodejs = Unicode(help=_('path to nodejs executable')).tag(config=True)
    node_roots = List_(trait=Any_(), default_value=[], help=_('absolute paths in which to seek node_modules')).tag(config=True)
    extra_node_roots = List_(trait=Any_(), default_value=[], help=_('additional absolute paths to seek node_modules first')).tag(config=True)

    def find_node_module(self, *path_frag):
        """look through the node_module roots to find the given node module"""
        all_roots = self.extra_node_roots + self.node_roots
        found = None
        for candidate_root in all_roots:
            candidate = pathlib.Path(candidate_root, 'node_modules', *path_frag)
            self.log.debug('Checking for %s', candidate)
            if candidate.exists():
                found = str(candidate)
                break
        if found is None:
            self.log.debug('{} not found in node_modules of {}'.format(pathlib.Path(*path_frag), all_roots))
        return found

    @default('nodejs')
    def _default_nodejs(self):
        return shutil.which('node') or shutil.which('nodejs') or shutil.which('nodejs.exe')

    @lru_cache(maxsize=1)
    def _npm_prefix(self, npm: Text):
        try:
            return subprocess.run([npm, 'prefix', '-g'], check=True, capture_output=True).stdout.decode('utf-8').strip()
        except Exception as e:
            self.log.warn(f'Could not determine npm prefix: {e}')

    @default('node_roots')
    def _default_node_roots(self):
        """get the "usual suspects" for where `node_modules` may be found

        - where this was launch (usually the same as NotebookApp.notebook_dir)
        - the JupyterLab staging folder (if available)
        - wherever conda puts it
        - wherever some other conventions put it
        """
        roots = [pathlib.Path.cwd()]
        try:
            from jupyterlab import commands
            roots += [pathlib.Path(commands.get_app_dir()) / 'staging']
        except ImportError:
            pass
        roots += [pathlib.Path(sys.prefix) / 'lib']
        roots += [pathlib.Path(sys.prefix)]
        npm = shutil.which('npm')
        if npm:
            prefix = self._npm_prefix(npm)
            if prefix:
                roots += [pathlib.Path(prefix) / 'lib', pathlib.Path(prefix)]
        return roots