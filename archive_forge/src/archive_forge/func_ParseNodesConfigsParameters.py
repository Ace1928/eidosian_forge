from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def ParseNodesConfigsParameters(nodes_configs):
    requested_node_types = [config['type'] for config in nodes_configs]
    duplicated_types = FindDuplicatedTypes(requested_node_types)
    if duplicated_types:
        raise InvalidNodeConfigsProvidedError('types: {} provided more than once.'.format(duplicated_types))
    return [NodeTypeConfig(config['type'], config['count'], config.get('custom-core-count', 0)) for config in nodes_configs]