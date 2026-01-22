from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def _lookup_ingress_egress_firewall_policy_ids(self, firewall_group):
    """
        Transforms firewall_group dict IN-PLACE. Takes the value of the keys
        egress_firewall_policy and ingress_firewall_policy, looks up the
        policy ids and maps them to egress_firewall_policy_id and
        ingress_firewall_policy_id. Old keys which were used for the lookup
        are deleted.

        :param dict firewall_group: firewall group dict
        :raises: DuplicateResource on multiple matches
        :raises: ResourceNotFound if a firewall policy is not found
        """
    for key in ('egress_firewall_policy', 'ingress_firewall_policy'):
        if key not in firewall_group:
            continue
        if firewall_group[key] is None:
            val = None
        else:
            val = self.network.find_firewall_policy(firewall_group[key], ignore_missing=False)['id']
        firewall_group[key + '_id'] = val
        del firewall_group[key]