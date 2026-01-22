from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def _get_enterprise_id(self):
    """
        Returns the identifier of the logged user's enterprise.
        """
    return self.connection.cache['enterprise'].findtext('id')