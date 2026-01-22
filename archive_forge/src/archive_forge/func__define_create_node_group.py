from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def _define_create_node_group(self, xml_loc, loc, group_name=None):
    """
        Search for a group where to create the node.

        If we can not find any group, create it into argument 'location'
        """
    if not group_name:
        group_name = NodeGroup.DEFAULT_GROUP_NAME
    groups_link = get_href(xml_loc, 'virtualappliances')
    groups_hdr = {'Accept': self.VAPPS_MIME_TYPE}
    vapps_element = self.connection.request(groups_link, headers=groups_hdr).object
    target_group = None
    for vapp in vapps_element.findall('virtualAppliance'):
        if vapp.findtext('name') == group_name:
            uri_vapp = get_href(vapp, 'edit')
            return NodeGroup(self, vapp.findtext('name'), uri=uri_vapp)
    if target_group is None:
        return self.ex_create_group(group_name, loc)