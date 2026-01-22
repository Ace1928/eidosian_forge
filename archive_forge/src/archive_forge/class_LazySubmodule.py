from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import subprocess
import importlib
import pkg_resources
import threading
from subprocess import check_output
from dataclasses import dataclass
from typing import Optional
from fileio import File, PathIO, PathIOLike
from lazyops.envs import logger
from lazyops.envs import LazyEnv
@dataclass
class LazySubmodule:
    name: str
    repo: Optional[str] = None
    module: Optional[str] = None
    module_path: Optional[str] = None
    init: Optional[bool] = False

    @property
    def main_module(self):
        return importlib.util.find_spec(self.name)

    @property
    def main_module_path(self):
        if not self.main_module:
            return None
        return self.main_module.submodule_search_locations[0]

    @property
    def submodule_name(self):
        if not self.module:
            return self.name
        if not self.module_path:
            return self.module + '.' + self.name
        if self.name in self.module_path:
            return self.module_path
        return self.module_path + '.' + self.name

    @property
    def submodule_namepath(self):
        return '/'.join(self.submodule_name.split('.'))

    @property
    def submodule_path(self):
        if not self.main_module_path:
            return None
        return File.join(self.main_module_path, self.submodule_namepath)

    @property
    def has_initialized(self):
        return bool(self.submodule_name in sys.modules)

    def lazy_import(self):
        if self.submodule_name and self.has_initialized:
            return
        if not File.exists(self.submodule_path) and self.repo and self.init:
            clone_repo(self.repo, path=self.submodule_path, abls=True)
        if File.exists(self.submodule_path):
            sys.modules[self.submodule_name] = importlib.import_module(self.submodule_name)
            logger.info(f'Initialized Submodule: {self.submodule_name}')
        else:
            logger.warn(f'Failed to Initilize Submodule: {self.submodule_name}')