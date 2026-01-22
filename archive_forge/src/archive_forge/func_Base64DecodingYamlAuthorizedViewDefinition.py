from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def Base64DecodingYamlAuthorizedViewDefinition(yaml_authorized_view):
    """Apply base64 decoding to all binary fields in the authorized view definition in YAML format."""
    if not yaml_authorized_view or 'subsetView' not in yaml_authorized_view:
        return yaml_authorized_view
    yaml_subset_view = yaml_authorized_view['subsetView']
    if 'rowPrefixes' in yaml_subset_view:
        yaml_subset_view['rowPrefixes'] = [Base64ToUtf8(s) for s in yaml_subset_view.get('rowPrefixes', [])]
    if 'familySubsets' in yaml_subset_view:
        for family_name, family_subset in yaml_subset_view['familySubsets'].items():
            if 'qualifiers' in family_subset:
                family_subset['qualifiers'] = [Base64ToUtf8(s) for s in family_subset.get('qualifiers', [])]
            if 'qualifierPrefixes' in family_subset:
                family_subset['qualifierPrefixes'] = [Base64ToUtf8(s) for s in family_subset.get('qualifierPrefixes', [])]
            yaml_subset_view['familySubsets'][family_name] = family_subset
    return yaml_authorized_view