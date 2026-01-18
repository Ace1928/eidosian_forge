from os_ken.base import app_manager
from os_ken.services.protocols.vrrp import event as vrrp_event
def vrrp_list(app, instance_name=None):
    """list instances.
    returns EventVRRPListReply([VRRPInstance]).
    """
    list_request = vrrp_event.EventVRRPListRequest(instance_name)
    list_request.dst = vrrp_event.VRRP_MANAGER_NAME
    return app.send_request(list_request)