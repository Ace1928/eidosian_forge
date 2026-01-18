from __future__ import absolute_import
import re
from ruamel.yaml.compat import string_types, _DEFAULT_YAML_VERSION  # NOQA
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode  # NOQA
from ruamel.yaml.util import RegExp  # NOQA
@property
def versioned_resolver(self):
    """
        select the resolver based on the version we are parsing
        """
    version = self.processing_version
    if version not in self._version_implicit_resolver:
        for x in implicit_resolvers:
            if version in x[0]:
                self.add_version_implicit_resolver(version, x[1], x[2], x[3])
    return self._version_implicit_resolver[version]