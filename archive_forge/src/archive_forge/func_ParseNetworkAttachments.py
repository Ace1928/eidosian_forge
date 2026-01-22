from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_printer
import six
def ParseNetworkAttachments(self, location, project, network_attachment):
    """Parse network attachments in flag to create network list."""
    networks = []
    for net in network_attachment:
        power_network = resources.REGISTRY.Parse(net, params={'projectsId': project.Name(), 'locationsId': location.Name()}, collection='marketplacesolutions.projects.locations.powerNetworks').RelativeName()
        networks.append(self.messages.NetworkAttachment(powerNetwork=power_network))
    return networks