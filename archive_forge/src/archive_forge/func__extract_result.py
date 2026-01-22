from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
@staticmethod
def _extract_result(details):
    if details is not None:
        return details.to_dict(computed=False)
    return {}