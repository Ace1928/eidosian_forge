from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class GCPProductOrganizationFallthrough(Fallthrough):
    """Falls through to the organization for the active GCP project."""
    _handled_fields = ['organization']

    def __init__(self):
        super(GCPProductOrganizationFallthrough, self).__init__('set the property [project] or provide the argument [--project] on the command line, using a Cloud Platform project with an associated Apigee organization')

    def _Call(self, parsed_args):
        return OrganizationFromGCPProduct()