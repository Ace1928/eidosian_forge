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
def with_collection_artifacts_manager(wrapped_method):
    """Inject an artifacts manager if not passed explicitly.

    This decorator constructs a ConcreteArtifactsManager and maintains
    the related temporary directory auto-cleanup around the target
    method invocation.
    """

    @functools.wraps(wrapped_method)
    def method_wrapper(*args, **kwargs):
        if 'artifacts_manager' in kwargs:
            return wrapped_method(*args, **kwargs)
        artifacts_manager_kwargs = {'validate_certs': context.CLIARGS.get('resolved_validate_certs', True)}
        keyring = context.CLIARGS.get('keyring', None)
        if keyring is not None:
            artifacts_manager_kwargs.update({'keyring': GalaxyCLI._resolve_path(keyring), 'required_signature_count': context.CLIARGS.get('required_valid_signature_count', None), 'ignore_signature_errors': context.CLIARGS.get('ignore_gpg_errors', None)})
        with ConcreteArtifactsManager.under_tmpdir(C.DEFAULT_LOCAL_TMP, **artifacts_manager_kwargs) as concrete_artifact_cm:
            kwargs['artifacts_manager'] = concrete_artifact_cm
            return wrapped_method(*args, **kwargs)
    return method_wrapper