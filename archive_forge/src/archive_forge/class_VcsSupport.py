import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
class VcsSupport:
    _registry: Dict[str, 'VersionControl'] = {}
    schemes = ['ssh', 'git', 'hg', 'bzr', 'sftp', 'svn']

    def __init__(self) -> None:
        urllib.parse.uses_netloc.extend(self.schemes)
        super().__init__()

    def __iter__(self) -> Iterator[str]:
        return self._registry.__iter__()

    @property
    def backends(self) -> List['VersionControl']:
        return list(self._registry.values())

    @property
    def dirnames(self) -> List[str]:
        return [backend.dirname for backend in self.backends]

    @property
    def all_schemes(self) -> List[str]:
        schemes: List[str] = []
        for backend in self.backends:
            schemes.extend(backend.schemes)
        return schemes

    def register(self, cls: Type['VersionControl']) -> None:
        if not hasattr(cls, 'name'):
            logger.warning('Cannot register VCS %s', cls.__name__)
            return
        if cls.name not in self._registry:
            self._registry[cls.name] = cls()
            logger.debug('Registered VCS backend: %s', cls.name)

    def unregister(self, name: str) -> None:
        if name in self._registry:
            del self._registry[name]

    def get_backend_for_dir(self, location: str) -> Optional['VersionControl']:
        """
        Return a VersionControl object if a repository of that type is found
        at the given directory.
        """
        vcs_backends = {}
        for vcs_backend in self._registry.values():
            repo_path = vcs_backend.get_repository_root(location)
            if not repo_path:
                continue
            logger.debug('Determine that %s uses VCS: %s', location, vcs_backend.name)
            vcs_backends[repo_path] = vcs_backend
        if not vcs_backends:
            return None
        inner_most_repo_path = max(vcs_backends, key=len)
        return vcs_backends[inner_most_repo_path]

    def get_backend_for_scheme(self, scheme: str) -> Optional['VersionControl']:
        """
        Return a VersionControl object or None.
        """
        for vcs_backend in self._registry.values():
            if scheme in vcs_backend.schemes:
                return vcs_backend
        return None

    def get_backend(self, name: str) -> Optional['VersionControl']:
        """
        Return a VersionControl object or None.
        """
        name = name.lower()
        return self._registry.get(name)