from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackNetworkOffering(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackNetworkOffering, self).__init__(module)
        self.returns = {'guestiptype': 'guest_ip_type', 'availability': 'availability', 'serviceofferingid': 'service_offering_id', 'networkrate': 'network_rate', 'maxconnections': 'max_connections', 'traffictype': 'traffic_type', 'isdefault': 'is_default', 'ispersistent': 'is_persistent', 'forvpc': 'for_vpc'}
        self.network_offering = None

    def get_service_offering_id(self):
        service_offering = self.module.params.get('service_offering')
        if not service_offering:
            return None
        args = {'issystem': True}
        service_offerings = self.query_api('listServiceOfferings', **args)
        if service_offerings:
            for s in service_offerings['serviceoffering']:
                if service_offering in [s['name'], s['id']]:
                    return s['id']
        self.fail_json(msg="Service offering '%s' not found" % service_offering)

    def get_network_offering(self):
        if self.network_offering:
            return self.network_offering
        args = {'name': self.module.params.get('name'), 'guestiptype': self.module.params.get('guest_type')}
        no = self.query_api('listNetworkOfferings', **args)
        if no:
            self.network_offering = no['networkoffering'][0]
        return self.network_offering

    def present(self):
        network_offering = self.get_network_offering()
        if not network_offering:
            network_offering = self.create_network_offering()
        if network_offering:
            network_offering = self.update_network_offering(network_offering=network_offering)
        return network_offering

    def create_network_offering(self):
        network_offering = None
        self.result['changed'] = True
        args = {'state': self.module.params.get('state'), 'displaytext': self.module.params.get('display_text'), 'guestiptype': self.module.params.get('guest_ip_type'), 'name': self.module.params.get('name'), 'supportedservices': self.module.params.get('supported_services'), 'traffictype': self.module.params.get('traffic_type'), 'availability': self.module.params.get('availability'), 'conservemode': self.module.params.get('conserve_mode'), 'details': self.module.params.get('details'), 'egressdefaultpolicy': self.module.params.get('egress_default_policy') == 'allow', 'ispersistent': self.module.params.get('persistent'), 'keepaliveenabled': self.module.params.get('keepalive_enabled'), 'maxconnections': self.module.params.get('max_connections'), 'networkrate': self.module.params.get('network_rate'), 'servicecapabilitylist': self.module.params.get('service_capabilities'), 'serviceofferingid': self.get_service_offering_id(), 'serviceproviderlist': self.module.params.get('service_providers'), 'specifyipranges': self.module.params.get('specify_ip_ranges'), 'specifyvlan': self.module.params.get('specify_vlan'), 'forvpc': self.module.params.get('for_vpc'), 'tags': self.module.params.get('tags'), 'domainid': self.module.params.get('domains'), 'zoneid': self.module.params.get('zones')}
        required_params = ['display_text', 'guest_ip_type', 'supported_services', 'service_providers']
        self.module.fail_on_missing_params(required_params=required_params)
        if not self.module.check_mode:
            res = self.query_api('createNetworkOffering', **args)
            network_offering = res['networkoffering']
        return network_offering

    def absent(self):
        network_offering = self.get_network_offering()
        if network_offering:
            self.result['changed'] = True
            if not self.module.check_mode:
                self.query_api('deleteNetworkOffering', id=network_offering['id'])
        return network_offering

    def update_network_offering(self, network_offering):
        tags = self.module.params.get('tags')
        domains = self.module.params.get('domains')
        zones = self.module.params.get('zones')
        args = {'id': network_offering['id'], 'state': self.module.params.get('state'), 'displaytext': self.module.params.get('display_text'), 'name': self.module.params.get('name'), 'availability': self.module.params.get('availability'), 'maxconnections': self.module.params.get('max_connections'), 'tags': ','.join(tags) if tags else None, 'domainid': ','.join(domains) if domains else None, 'zoneid': ','.join(zones) if zones else None}
        if args['state'] in ['enabled', 'disabled']:
            args['state'] = args['state'].title()
        else:
            del args['state']
        if self.has_changed(args, network_offering):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateNetworkOffering', **args)
                network_offering = res['networkoffering']
        return network_offering

    def get_result(self, resource):
        super(AnsibleCloudStackNetworkOffering, self).get_result(resource)
        if resource:
            self.result['egress_default_policy'] = 'allow' if resource.get('egressdefaultpolicy') else 'deny'
            tags = resource.get('tags')
            self.result['tags'] = tags.split(',') if tags else []
            zone_id = resource.get('zoneid')
            self.result['zones'] = zone_id.split(',') if zone_id else []
            domain_id = resource.get('domainid')
            self.result['domains'] = zone_id.split(',') if domain_id else []
        return self.result