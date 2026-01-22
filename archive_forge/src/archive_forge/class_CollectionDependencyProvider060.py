from __future__ import (absolute_import, division, print_function)
import functools
import typing as t
from ansible.galaxy.collection.gpg import get_signature_from_source
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import (
from ansible.module_utils.six import string_types
from ansible.utils.version import SemanticVersion, LooseVersion
class CollectionDependencyProvider060(CollectionDependencyProviderBase):

    def find_matches(self, identifier, requirements, incompatibilities):
        return [match for match in self._find_matches(list(requirements[identifier])) if not any((match.ver == incompat.ver for incompat in incompatibilities[identifier]))]

    def get_preference(self, resolution, candidates, information):
        return self._get_preference(candidates)