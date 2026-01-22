from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.exceptions import Error
from googlecloudsdk.core.resources import REGISTRY
class NetworkNotFoundError(Error):

    def __init__(self, network_id):
        super(NetworkNotFoundError, self).__init__("FAILED_PRECONDITION: The VMware Engine network `{network_id}` doesn't exist. Operation on the resource can't be fulfilled.".format(network_id=network_id))