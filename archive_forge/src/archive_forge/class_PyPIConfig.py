import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
class PyPIConfig(configparser.RawConfigParser):

    def __init__(self):
        """
        Load from ~/.pypirc
        """
        defaults = dict.fromkeys(['username', 'password', 'repository'], '')
        super().__init__(defaults)
        rc = os.path.join(os.path.expanduser('~'), '.pypirc')
        if os.path.exists(rc):
            self.read(rc)

    @property
    def creds_by_repository(self):
        sections_with_repositories = [section for section in self.sections() if self.get(section, 'repository').strip()]
        return dict(map(self._get_repo_cred, sections_with_repositories))

    def _get_repo_cred(self, section):
        repo = self.get(section, 'repository').strip()
        return (repo, Credential(self.get(section, 'username').strip(), self.get(section, 'password').strip()))

    def find_credential(self, url):
        """
        If the URL indicated appears to be a repository defined in this
        config, return the credential for that repository.
        """
        for repository, cred in self.creds_by_repository.items():
            if url.startswith(repository):
                return cred