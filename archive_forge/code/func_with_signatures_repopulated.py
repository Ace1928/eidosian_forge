from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections import namedtuple
from collections.abc import MutableSequence, MutableMapping
from glob import iglob
from urllib.parse import urlparse
from yaml import safe_load
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection import HAS_PACKAGING, PkgReq
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
def with_signatures_repopulated(self):
    """Populate a new Candidate instance with Galaxy signatures.
        :raises AnsibleAssertionError: If the supplied candidate is not sourced from a Galaxy-like index.
        """
    if self.type != 'galaxy':
        raise AnsibleAssertionError(f'Invalid collection type for {self!r}: unable to get signatures from a galaxy server.')
    signatures = self.src.get_collection_signatures(self.namespace, self.name, self.ver)
    return self.__class__(self.fqcn, self.ver, self.src, self.type, frozenset([*self.signatures, *signatures]))