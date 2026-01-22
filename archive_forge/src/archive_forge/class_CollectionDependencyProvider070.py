from __future__ import (absolute_import, division, print_function)
import functools
import typing as t
from ansible.galaxy.collection.gpg import get_signature_from_source
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import (
from ansible.module_utils.six import string_types
from ansible.utils.version import SemanticVersion, LooseVersion
class CollectionDependencyProvider070(CollectionDependencyProvider060):

    def get_preference(self, identifier, resolutions, candidates, information):
        return self._get_preference(list(candidates[identifier]))