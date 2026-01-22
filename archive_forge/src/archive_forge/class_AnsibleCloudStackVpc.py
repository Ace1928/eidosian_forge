from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackVpc(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVpc, self).__init__(module)
        self.returns = {'cidr': 'cidr', 'networkdomain': 'network_domain', 'redundantvpcrouter': 'redundant_vpc_router', 'distributedvpcrouter': 'distributed_vpc_router', 'regionlevelvpc': 'region_level_vpc', 'restartrequired': 'restart_required'}
        self.vpc = None

    def get_vpc_offering(self, key=None):
        vpc_offering = self.module.params.get('vpc_offering')
        args = {'state': 'Enabled'}
        if vpc_offering:
            args['name'] = vpc_offering
            fail_msg = 'VPC offering not found or not enabled: %s' % vpc_offering
        else:
            args['isdefault'] = True
            fail_msg = 'No enabled default VPC offering found'
        vpc_offerings = self.query_api('listVPCOfferings', **args)
        if vpc_offerings:
            for vo in vpc_offerings['vpcoffering']:
                if 'name' in args:
                    if args['name'] == vo['name']:
                        return self._get_by_key(key, vo)
                else:
                    return self._get_by_key(key, vo)
        self.module.fail_json(msg=fail_msg)

    def get_vpc(self):
        if self.vpc:
            return self.vpc
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'fetch_list': True}
        vpcs = self.query_api('listVPCs', **args)
        if vpcs:
            vpc_name = self.module.params.get('name')
            for v in vpcs:
                if vpc_name in [v['name'], v['displaytext'], v['id']]:
                    if self.vpc:
                        self.module.fail_json(msg='More than one VPC found with the provided identifyer: %s' % vpc_name)
                    else:
                        self.vpc = v
        return self.vpc

    def restart_vpc(self):
        self.result['changed'] = True
        vpc = self.get_vpc()
        if vpc and (not self.module.check_mode):
            args = {'id': vpc['id'], 'cleanup': self.module.params.get('clean_up')}
            res = self.query_api('restartVPC', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'vpc')
        return vpc

    def present_vpc(self):
        vpc = self.get_vpc()
        if not vpc:
            vpc = self._create_vpc(vpc)
        else:
            vpc = self._update_vpc(vpc)
        if vpc:
            vpc = self.ensure_tags(resource=vpc, resource_type='Vpc')
        return vpc

    def _create_vpc(self, vpc):
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'networkdomain': self.module.params.get('network_domain'), 'vpcofferingid': self.get_vpc_offering(key='id'), 'cidr': self.module.params.get('cidr'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'start': self.module.params.get('state') != 'stopped'}
        self.result['diff']['after'] = args
        if not self.module.check_mode:
            res = self.query_api('createVPC', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vpc = self.poll_job(res, 'vpc')
        return vpc

    def _update_vpc(self, vpc):
        args = {'id': vpc['id'], 'displaytext': self.module.params.get('display_text')}
        if self.has_changed(args, vpc):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateVPC', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    vpc = self.poll_job(res, 'vpc')
        return vpc

    def absent_vpc(self):
        vpc = self.get_vpc()
        if vpc:
            self.result['changed'] = True
            self.result['diff']['before'] = vpc
            if not self.module.check_mode:
                res = self.query_api('deleteVPC', id=vpc['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'vpc')
        return vpc