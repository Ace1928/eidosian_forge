from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackInstance(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackInstance, self).__init__(module)
        self.returns = {'group': 'group', 'hypervisor': 'hypervisor', 'instancename': 'instance_name', 'publicip': 'public_ip', 'passwordenabled': 'password_enabled', 'password': 'password', 'serviceofferingname': 'service_offering', 'isoname': 'iso', 'templatename': 'template', 'templatedisplaytext': 'template_display_text', 'keypair': 'ssh_key', 'hostname': 'host'}
        self.instance = None
        self.template = None
        self.iso = None

    def get_service_offering_id(self):
        service_offering = self.module.params.get('service_offering')
        service_offerings = self.query_api('listServiceOfferings')
        if service_offerings:
            if not service_offering:
                return service_offerings['serviceoffering'][0]['id']
            for s in service_offerings['serviceoffering']:
                if service_offering in [s['name'], s['id']]:
                    return s['id']
        self.fail_json(msg="Service offering '%s' not found" % service_offering)

    def get_host_id(self):
        host_name = self.module.params.get('host')
        if not host_name:
            return None
        args = {'type': 'routing', 'zoneid': self.get_zone(key='id')}
        hosts = self.query_api('listHosts', **args)
        if hosts:
            for h in hosts['host']:
                if host_name in [h['name'], h['id']]:
                    return h['id']
        self.fail_json(msg="Host '%s' not found" % host_name)

    def get_cluster_id(self):
        cluster_name = self.module.params.get('cluster')
        if not cluster_name:
            return None
        args = {'zoneid': self.get_zone(key='id')}
        clusters = self.query_api('listClusters', **args)
        if clusters:
            for c in clusters['cluster']:
                if cluster_name in [c['name'], c['id']]:
                    return c['id']
        self.fail_json(msg="Cluster '%s' not found" % cluster_name)

    def get_pod_id(self):
        pod_name = self.module.params.get('pod')
        if not pod_name:
            return None
        args = {'zoneid': self.get_zone(key='id')}
        pods = self.query_api('listPods', **args)
        if pods:
            for p in pods['pod']:
                if pod_name in [p['name'], p['id']]:
                    return p['id']
        self.fail_json(msg="Pod '%s' not found" % pod_name)

    def get_template_or_iso(self, key=None):
        template = self.module.params.get('template')
        iso = self.module.params.get('iso')
        if not template and (not iso):
            return None
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'isrecursive': True, 'fetch_list': True}
        if template:
            if self.template:
                return self._get_by_key(key, self.template)
            rootdisksize = self.module.params.get('root_disk_size')
            args['templatefilter'] = self.module.params.get('template_filter')
            args['fetch_list'] = True
            templates = self.query_api('listTemplates', **args)
            if templates:
                for t in templates:
                    if template in [t.get('displaytext', None), t['name'], t['id']]:
                        if rootdisksize and t['size'] > rootdisksize * 1024 ** 3:
                            continue
                        self.template = t
                        return self._get_by_key(key, self.template)
            if rootdisksize:
                more_info = ' (with size <= %s)' % rootdisksize
            else:
                more_info = ''
            self.module.fail_json(msg="Template '%s' not found%s" % (template, more_info))
        elif iso:
            if self.iso:
                return self._get_by_key(key, self.iso)
            args['isofilter'] = self.module.params.get('template_filter')
            args['fetch_list'] = True
            isos = self.query_api('listIsos', **args)
            if isos:
                for i in isos:
                    if iso in [i['displaytext'], i['name'], i['id']]:
                        self.iso = i
                        return self._get_by_key(key, self.iso)
            self.module.fail_json(msg="ISO '%s' not found" % iso)

    def get_instance(self):
        instance = self.instance
        if not instance:
            instance_name = self.get_or_fallback('name', 'display_name')
            args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'fetch_list': True}
            instances = self.query_api('listVirtualMachines', **args)
            if instances:
                for v in instances:
                    if instance_name.lower() in [v['name'].lower(), v['displayname'].lower(), v['id']]:
                        self.instance = v
                        break
        return self.instance

    def _get_instance_user_data(self, instance):
        if 'userdata' in instance:
            return instance['userdata']
        user_data = ''
        if self.get_user_data() is not None and instance.get('id'):
            res = self.query_api('getVirtualMachineUserData', virtualmachineid=instance['id'])
            user_data = res['virtualmachineuserdata'].get('userdata', '')
        return user_data

    def get_iptonetwork_mappings(self):
        network_mappings = self.module.params.get('ip_to_networks')
        if network_mappings is None:
            return
        if network_mappings and self.module.params.get('networks'):
            self.module.fail_json(msg='networks and ip_to_networks are mutually exclusive.')
        network_names = [n['network'] for n in network_mappings]
        ids = self.get_network_ids(network_names)
        res = []
        for i, data in enumerate(network_mappings):
            res.append(dict(networkid=ids[i], **data))
        return res

    def get_ssh_keypair(self, key=None, name=None, fail_on_missing=True):
        ssh_key_name = name or self.module.params.get('ssh_key')
        if ssh_key_name is None:
            return
        args = {'domainid': self.get_domain('id'), 'account': self.get_account('name'), 'projectid': self.get_project('id'), 'name': ssh_key_name}
        ssh_key_pairs = self.query_api('listSSHKeyPairs', **args)
        if 'sshkeypair' in ssh_key_pairs:
            return self._get_by_key(key=key, my_dict=ssh_key_pairs['sshkeypair'][0])
        elif fail_on_missing:
            self.module.fail_json(msg='SSH key not found: %s' % ssh_key_name)

    def ssh_key_has_changed(self):
        ssh_key_name = self.module.params.get('ssh_key')
        if ssh_key_name is None:
            return False
        param_ssh_key_fp = self.get_ssh_keypair(key='fingerprint')
        instance_ssh_key_name = self.instance.get('keypair')
        if instance_ssh_key_name is None:
            return True
        instance_ssh_key_fp = self.get_ssh_keypair(key='fingerprint', name=instance_ssh_key_name, fail_on_missing=False)
        if not instance_ssh_key_fp:
            return True
        if instance_ssh_key_fp != param_ssh_key_fp:
            return True
        return False

    def security_groups_has_changed(self):
        security_groups = self.module.params.get('security_groups')
        if security_groups is None:
            return False
        security_groups = [s.lower() for s in security_groups]
        instance_security_groups = self.instance.get('securitygroup') or []
        instance_security_group_names = []
        for instance_security_group in instance_security_groups:
            if instance_security_group['name'].lower() not in security_groups:
                return True
            else:
                instance_security_group_names.append(instance_security_group['name'].lower())
        for security_group in security_groups:
            if security_group not in instance_security_group_names:
                return True
        return False

    def get_network_ids(self, network_names=None):
        if network_names is None:
            network_names = self.module.params.get('networks')
        if not network_names:
            return None
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'fetch_list': True}
        networks = self.query_api('listNetworks', **args)
        if not networks:
            self.module.fail_json(msg='No networks available')
        network_ids = []
        network_displaytexts = []
        for network_name in network_names:
            for n in networks:
                if network_name in [n['displaytext'], n['name'], n['id']]:
                    network_ids.append(n['id'])
                    network_displaytexts.append(n['name'])
                    break
        if len(network_ids) != len(network_names):
            self.module.fail_json(msg='Could not find all networks, networks list found: %s' % network_displaytexts)
        return network_ids

    def present_instance(self, start_vm=True):
        instance = self.get_instance()
        if not instance:
            instance = self.deploy_instance(start_vm=start_vm)
        else:
            instance = self.recover_instance(instance=instance)
            instance = self.update_instance(instance=instance, start_vm=start_vm)
        if instance:
            instance = self.ensure_tags(resource=instance, resource_type='UserVm')
            self.instance = instance
        return instance

    def get_user_data(self):
        user_data = self.module.params.get('user_data')
        if user_data is not None:
            user_data = to_text(base64.b64encode(to_bytes(user_data)))
        return user_data

    def get_details(self):
        details = self.module.params.get('details')
        cpu = self.module.params.get('cpu')
        cpu_speed = self.module.params.get('cpu_speed')
        memory = self.module.params.get('memory')
        if any([cpu, cpu_speed, memory]):
            if details is None:
                details = {}
            if cpu:
                details['cpuNumber'] = cpu
            if cpu_speed:
                details['cpuSpeed'] = cpu_speed
            if memory:
                details['memory'] = memory
        return details

    def deploy_instance(self, start_vm=True):
        self.result['changed'] = True
        networkids = self.get_network_ids()
        if networkids is not None:
            networkids = ','.join(networkids)
        args = {}
        args['templateid'] = self.get_template_or_iso(key='id')
        if not args['templateid']:
            self.module.fail_json(msg='Template or ISO is required.')
        args['zoneid'] = self.get_zone(key='id')
        args['serviceofferingid'] = self.get_service_offering_id()
        args['account'] = self.get_account(key='name')
        args['domainid'] = self.get_domain(key='id')
        args['projectid'] = self.get_project(key='id')
        args['diskofferingid'] = self.get_disk_offering(key='id')
        args['networkids'] = networkids
        args['iptonetworklist'] = self.get_iptonetwork_mappings()
        args['userdata'] = self.get_user_data()
        args['keyboard'] = self.module.params.get('keyboard')
        args['ipaddress'] = self.module.params.get('ip_address')
        args['ip6address'] = self.module.params.get('ip6_address')
        args['name'] = self.module.params.get('name')
        args['displayname'] = self.get_or_fallback('display_name', 'name')
        args['group'] = self.module.params.get('group')
        args['keypair'] = self.get_ssh_keypair(key='name')
        args['size'] = self.module.params.get('disk_size')
        args['startvm'] = start_vm
        args['rootdisksize'] = self.module.params.get('root_disk_size')
        args['affinitygroupnames'] = self.module.params.get('affinity_groups')
        args['details'] = self.get_details()
        args['securitygroupnames'] = self.module.params.get('security_groups')
        args['hostid'] = self.get_host_id()
        args['clusterid'] = self.get_cluster_id()
        args['podid'] = self.get_pod_id()
        template_iso = self.get_template_or_iso()
        if 'hypervisor' not in template_iso:
            args['hypervisor'] = self.get_hypervisor()
        instance = None
        if not self.module.check_mode:
            instance = self.query_api('deployVirtualMachine', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                instance = self.poll_job(instance, 'virtualmachine')
        return instance

    def update_instance(self, instance, start_vm=True):
        args_service_offering = {'id': instance['id']}
        if self.module.params.get('service_offering'):
            args_service_offering['serviceofferingid'] = self.get_service_offering_id()
        service_offering_changed = self.has_changed(args_service_offering, instance)
        args_instance_update = {'id': instance['id'], 'userdata': self.get_user_data()}
        instance['userdata'] = self._get_instance_user_data(instance)
        args_instance_update['ostypeid'] = self.get_os_type(key='id')
        if self.module.params.get('group'):
            args_instance_update['group'] = self.module.params.get('group')
        if self.module.params.get('display_name'):
            args_instance_update['displayname'] = self.module.params.get('display_name')
        instance_changed = self.has_changed(args_instance_update, instance)
        ssh_key_changed = self.ssh_key_has_changed()
        security_groups_changed = self.security_groups_has_changed()
        args_volume_update = {}
        root_disk_size = self.module.params.get('root_disk_size')
        root_disk_size_changed = False
        if root_disk_size is not None:
            args = {'type': 'ROOT', 'virtualmachineid': instance['id'], 'account': instance.get('account'), 'domainid': instance.get('domainid'), 'projectid': instance.get('projectid')}
            res = self.query_api('listVolumes', **args)
            [volume] = res['volume']
            size = volume['size'] >> 30
            args_volume_update['id'] = volume['id']
            args_volume_update['size'] = root_disk_size
            shrinkok = self.module.params.get('allow_root_disk_shrink')
            if shrinkok:
                args_volume_update['shrinkok'] = shrinkok
            root_disk_size_changed = root_disk_size != size
        changed = [service_offering_changed, instance_changed, security_groups_changed, ssh_key_changed, root_disk_size_changed]
        if any(changed):
            force = self.module.params.get('force')
            instance_state = instance['state'].lower()
            if instance_state == 'stopped' or force:
                self.result['changed'] = True
                if not self.module.check_mode:
                    instance = self.stop_instance()
                    instance = self.poll_job(instance, 'virtualmachine')
                    self.instance = instance
                    if service_offering_changed:
                        res = self.query_api('changeServiceForVirtualMachine', **args_service_offering)
                        instance = res['virtualmachine']
                        self.instance = instance
                    if instance_changed or security_groups_changed:
                        if security_groups_changed:
                            args_instance_update['securitygroupnames'] = ','.join(self.module.params.get('security_groups'))
                        res = self.query_api('updateVirtualMachine', **args_instance_update)
                        instance = res['virtualmachine']
                        self.instance = instance
                    if ssh_key_changed:
                        args_ssh_key = {}
                        args_ssh_key['id'] = instance['id']
                        args_ssh_key['projectid'] = self.get_project(key='id')
                        args_ssh_key['keypair'] = self.module.params.get('ssh_key')
                        instance = self.query_api('resetSSHKeyForVirtualMachine', **args_ssh_key)
                        instance = self.poll_job(instance, 'virtualmachine')
                        self.instance = instance
                    if root_disk_size_changed:
                        async_result = self.query_api('resizeVolume', **args_volume_update)
                        self.poll_job(async_result, 'volume')
                    if instance_state == 'running' and start_vm:
                        instance = self.start_instance()
            else:
                self.module.warn("Changes won't be applied to running instances. Use force=true to allow the instance %s to be stopped/started." % instance['name'])
        host_changed = all([instance['state'].lower() in ['starting', 'running'], instance.get('hostname') is not None, self.module.params.get('host') is not None, self.module.params.get('host') != instance.get('hostname')])
        if host_changed:
            self.result['changed'] = True
            args_host = {'virtualmachineid': instance['id'], 'hostid': self.get_host_id()}
            if not self.module.check_mode:
                res = self.query_api('migrateVirtualMachine', **args_host)
                instance = self.poll_job(res, 'virtualmachine')
        return instance

    def recover_instance(self, instance):
        if instance['state'].lower() in ['destroying', 'destroyed']:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('recoverVirtualMachine', id=instance['id'])
                instance = res['virtualmachine']
        return instance

    def absent_instance(self):
        instance = self.get_instance()
        if instance:
            if instance['state'].lower() not in ['expunging', 'destroying', 'destroyed']:
                self.result['changed'] = True
                if not self.module.check_mode:
                    res = self.query_api('destroyVirtualMachine', id=instance['id'])
                    poll_async = self.module.params.get('poll_async')
                    if poll_async:
                        instance = self.poll_job(res, 'virtualmachine')
        return instance

    def expunge_instance(self):
        instance = self.get_instance()
        if instance:
            res = {}
            if instance['state'].lower() in ['destroying', 'destroyed']:
                self.result['changed'] = True
                if not self.module.check_mode:
                    res = self.query_api('destroyVirtualMachine', id=instance['id'], expunge=True)
            elif instance['state'].lower() not in ['expunging']:
                self.result['changed'] = True
                if not self.module.check_mode:
                    res = self.query_api('destroyVirtualMachine', id=instance['id'], expunge=True)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                res = self.poll_job(res, 'virtualmachine')
        return instance

    def stop_instance(self):
        instance = self.get_instance()
        if instance:
            if instance['state'].lower() in ['stopping', 'stopped']:
                return instance
            if instance['state'].lower() in ['starting', 'running']:
                self.result['changed'] = True
                if not self.module.check_mode:
                    instance = self.query_api('stopVirtualMachine', id=instance['id'])
                    poll_async = self.module.params.get('poll_async')
                    if poll_async:
                        instance = self.poll_job(instance, 'virtualmachine')
        return instance

    def start_instance(self):
        instance = self.get_instance()
        if instance:
            if instance['state'].lower() in ['starting', 'running']:
                return instance
            if instance['state'].lower() in ['stopped', 'stopping']:
                self.result['changed'] = True
                if not self.module.check_mode:
                    args = {'id': instance['id'], 'hostid': self.get_host_id()}
                    instance = self.query_api('startVirtualMachine', **args)
                    poll_async = self.module.params.get('poll_async')
                    if poll_async:
                        instance = self.poll_job(instance, 'virtualmachine')
        return instance

    def restart_instance(self):
        instance = self.get_instance()
        if instance:
            if instance['state'].lower() in ['running', 'starting']:
                self.result['changed'] = True
                if not self.module.check_mode:
                    instance = self.query_api('rebootVirtualMachine', id=instance['id'])
                    poll_async = self.module.params.get('poll_async')
                    if poll_async:
                        instance = self.poll_job(instance, 'virtualmachine')
            elif instance['state'].lower() in ['stopping', 'stopped']:
                instance = self.start_instance()
        return instance

    def restore_instance(self):
        instance = self.get_instance()
        self.result['changed'] = True
        if instance:
            args = {}
            args['templateid'] = self.get_template_or_iso(key='id')
            args['virtualmachineid'] = instance['id']
            res = self.query_api('restoreVirtualMachine', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                instance = self.poll_job(res, 'virtualmachine')
        return instance

    def get_result(self, resource):
        super(AnsibleCloudStackInstance, self).get_result(resource)
        if resource:
            self.result['user_data'] = self._get_instance_user_data(resource)
            if 'securitygroup' in resource:
                security_groups = []
                for securitygroup in resource['securitygroup']:
                    security_groups.append(securitygroup['name'])
                self.result['security_groups'] = security_groups
            if 'affinitygroup' in resource:
                affinity_groups = []
                for affinitygroup in resource['affinitygroup']:
                    affinity_groups.append(affinitygroup['name'])
                self.result['affinity_groups'] = affinity_groups
            if 'nic' in resource:
                for nic in resource['nic']:
                    if nic['isdefault']:
                        if 'ipaddress' in nic:
                            self.result['default_ip'] = nic['ipaddress']
                        if 'ip6address' in nic:
                            self.result['default_ip6'] = nic['ip6address']
        return self.result