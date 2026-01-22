from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec
class AnsibleCloudStackInstanceInfo(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackInstanceInfo, self).__init__(module)
        self.returns = {'group': 'group', 'hypervisor': 'hypervisor', 'instancename': 'instance_name', 'publicip': 'public_ip', 'passwordenabled': 'password_enabled', 'password': 'password', 'serviceofferingname': 'service_offering', 'isoname': 'iso', 'templatename': 'template', 'keypair': 'ssh_key', 'hostname': 'host'}

    def get_host(self, key=None):
        host = self.module.params.get('host')
        if not host:
            return
        args = {'fetch_list': True}
        res = self.query_api('listHosts', **args)
        if res:
            for h in res:
                if host.lower() in [h['id'], h['ipaddress'], h['name'].lower()]:
                    return self._get_by_key(key, h)
        self.fail_json(msg='Host not found: %s' % host)

    def get_instances(self):
        instance_name = self.module.params.get('name')
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'hostid': self.get_host(key='id'), 'fetch_list': True}
        instances = self.query_api('listVirtualMachines', **args)
        if not instance_name:
            return instances or []
        if instances:
            for v in instances:
                if instance_name.lower() in [v['name'].lower(), v['displayname'].lower(), v['id']]:
                    return [v]
        return []

    def get_volumes(self, instance):
        volume_details = []
        if instance:
            args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'virtualmachineid': instance['id'], 'fetch_list': True}
            volumes = self.query_api('listVolumes', **args)
            if volumes:
                for vol in volumes:
                    volume_details.append({'size': vol['size'], 'type': vol['type'], 'name': vol['name']})
        return volume_details

    def run(self):
        instances = self.get_instances()
        if self.module.params.get('name') and (not instances):
            self.module.fail_json(msg='Instance not found: %s' % self.module.params.get('name'))
        return {'instances': [self.update_result(resource) for resource in instances]}

    def update_result(self, resource, result=None):
        result = super(AnsibleCloudStackInstanceInfo, self).update_result(resource, result)
        if resource:
            if 'securitygroup' in resource:
                security_groups = []
                for securitygroup in resource['securitygroup']:
                    security_groups.append(securitygroup['name'])
                result['security_groups'] = security_groups
            if 'affinitygroup' in resource:
                affinity_groups = []
                for affinitygroup in resource['affinitygroup']:
                    affinity_groups.append(affinitygroup['name'])
                result['affinity_groups'] = affinity_groups
            if 'nic' in resource:
                for nic in resource['nic']:
                    if nic['isdefault'] and 'ipaddress' in nic:
                        result['default_ip'] = nic['ipaddress']
                result['nic'] = resource['nic']
            volumes = self.get_volumes(instance=resource)
            if volumes:
                result['volumes'] = volumes
        return result