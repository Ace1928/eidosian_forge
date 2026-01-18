import logging
import os
from oslo_utils import strutils
from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def vif_detach(self, node_ident, vif_id, os_ironic_api_version=None, global_request_id=None):
    """Detach VIF from a given node.

        :param node_ident: The UUID or Name of the node.
        :param vif_id: The UUID or Name of the VIF to detach.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.
        """
    path = '%s/vifs/%s' % (node_ident, vif_id)
    self.delete(path, os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)