from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
def _AttributeName(self, parameter_name):
    """Helper function to get the corresponding attribute for a parameter."""
    return self.resource_info.resource_spec.AttributeName(parameter_name)