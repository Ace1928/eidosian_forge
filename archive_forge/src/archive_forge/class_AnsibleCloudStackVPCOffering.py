from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackVPCOffering(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVPCOffering, self).__init__(module)
        self.returns = {'serviceofferingid': 'service_offering_id', 'isdefault': 'is_default', 'distributedvpcrouter': 'distributed', 'supportsregionLevelvpc': 'region_level'}
        self.vpc_offering = None

    def get_vpc_offering(self):
        if self.vpc_offering:
            return self.vpc_offering
        args = {'name': self.module.params.get('name')}
        vo = self.query_api('listVPCOfferings', **args)
        if vo:
            for vpc_offer in vo['vpcoffering']:
                if args['name'] == vpc_offer['name']:
                    self.vpc_offering = vpc_offer
        return self.vpc_offering

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

    def create_or_update(self):
        vpc_offering = self.get_vpc_offering()
        if not vpc_offering:
            vpc_offering = self.create_vpc_offering()
        return self.update_vpc_offering(vpc_offering)

    def create_vpc_offering(self):
        vpc_offering = None
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'state': self.module.params.get('state'), 'displaytext': self.module.params.get('display_text'), 'supportedservices': self.module.params.get('supported_services'), 'serviceproviderlist': self.module.params.get('service_providers'), 'serviceofferingid': self.get_service_offering_id(), 'servicecapabilitylist': self.module.params.get('service_capabilities')}
        required_params = ['display_text', 'supported_services']
        self.module.fail_on_missing_params(required_params=required_params)
        if not self.module.check_mode:
            res = self.query_api('createVPCOffering', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vpc_offering = self.poll_job(res, 'vpcoffering')
        return vpc_offering

    def delete_vpc_offering(self):
        vpc_offering = self.get_vpc_offering()
        if vpc_offering:
            self.result['changed'] = True
            args = {'id': vpc_offering['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteVPCOffering', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    vpc_offering = self.poll_job(res, 'vpcoffering')
        return vpc_offering

    def update_vpc_offering(self, vpc_offering):
        if not vpc_offering:
            return vpc_offering
        args = {'id': vpc_offering['id'], 'state': self.module.params.get('state'), 'name': self.module.params.get('name'), 'displaytext': self.module.params.get('display_text')}
        if args['state'] in ['enabled', 'disabled']:
            args['state'] = args['state'].title()
        else:
            del args['state']
        if self.has_changed(args, vpc_offering):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateVPCOffering', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    vpc_offering = self.poll_job(res, 'vpcoffering')
        return vpc_offering