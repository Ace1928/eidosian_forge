from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def LoadSoleTenantConfigFromNodeAffinityYaml(affinities_yaml, messages):
    """Loads json/yaml node affinities from yaml file contents."""
    if not affinities_yaml:
        raise Error('No node affinity labels specified. You must specify at least one label to create a sole tenancy instance.')
    if not yaml.list_like(affinities_yaml):
        raise Error('Node affinities must be specified as JSON/YAML list')
    node_affinities = []
    for affinity in affinities_yaml:
        node_affinity = None
        if not affinity:
            raise Error('Empty list item in JSON/YAML file.')
        try:
            node_affinity = encoding.PyValueToMessage(messages.NodeAffinity, affinity)
        except Exception as e:
            raise Error(e)
        if not node_affinity.key:
            raise Error('A key must be specified for every node affinity label.')
        if node_affinity.all_unrecognized_fields():
            raise Error('Key [{0}] has invalid field formats for: {1}'.format(node_affinity.key, node_affinity.all_unrecognized_fields()))
        node_affinities.append(node_affinity)
    return messages.SoleTenantConfig(nodeAffinities=node_affinities)