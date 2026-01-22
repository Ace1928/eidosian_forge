from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _MaxPodsConstraint(self, args):
    kwargs = {'maxPodsPerNode': flags.GetMaxPodsPerNode(args)}
    return self._messages.GoogleCloudGkemulticloudV1MaxPodsConstraint(**kwargs) if any(kwargs.values()) else None