from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import argparse
import functools
import json
import os.path
import pathlib
import re
import shutil
import sys
import textwrap
import time
import typing as t
from dataclasses import dataclass
from yaml.error import YAMLError
import ansible.constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.galaxy import Galaxy, get_collections_galaxy_meta_info
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.galaxy.collection import (
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.gpg import GPG_ERROR_MAP
from ansible.galaxy.dependency_resolution.dataclasses import Requirement
from ansible.galaxy.role import GalaxyRole
from ansible.galaxy.token import BasicAuthToken, GalaxyToken, KeycloakToken, NoTokenSentinel
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils import six
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.playbook.role.requirement import RoleRequirement
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
def add_download_options(self, parser, parents=None):
    download_parser = parser.add_parser('download', parents=parents, help='Download collections and their dependencies as a tarball for an offline install.')
    download_parser.set_defaults(func=self.execute_download)
    download_parser.add_argument('args', help='Collection(s)', metavar='collection', nargs='*')
    download_parser.add_argument('-n', '--no-deps', dest='no_deps', action='store_true', default=False, help="Don't download collection(s) listed as dependencies.")
    download_parser.add_argument('-p', '--download-path', dest='download_path', default='./collections', help='The directory to download the collections to.')
    download_parser.add_argument('-r', '--requirements-file', dest='requirements', help='A file containing a list of collections to be downloaded.')
    download_parser.add_argument('--pre', dest='allow_pre_release', action='store_true', help='Include pre-release versions. Semantic versioning pre-releases are ignored by default')