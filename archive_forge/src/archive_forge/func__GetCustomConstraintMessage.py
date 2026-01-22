from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.policy_intelligence import orgpolicy_simulator
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _GetCustomConstraintMessage():
    """Returns the organization custom constraint message."""
    return 'GoogleCloudOrgpolicy' + 'V2' + 'CustomConstraint'