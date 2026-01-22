from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def _to_nodeimage(self, template, driver, repo):
    """
        Generates the :class:`NodeImage` class.
        """
    identifier = template.findtext('id')
    name = template.findtext('name')
    url = get_href(template, 'edit')
    hdreqd = template.findtext('hdRequired')
    extra = {'repo': repo, 'url': url, 'hdrequired': hdreqd}
    return NodeImage(identifier, name, driver, extra)