from __future__ import absolute_import, division, print_function
import re
import time
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxLxcAnsible(ProxmoxAnsible):

    def content_check(self, node, ostemplate, template_store):
        return [True for cnt in self.proxmox_api.nodes(node).storage(template_store).content.get() if cnt['volid'] == ostemplate]

    def is_template_container(self, node, vmid):
        """Check if the specified container is a template."""
        proxmox_node = self.proxmox_api.nodes(node)
        config = getattr(proxmox_node, VZ_TYPE)(vmid).config.get()
        return config.get('template', False)

    def update_config(self, vmid, node, disk, cpus, memory, swap, **kwargs):
        if VZ_TYPE != 'lxc':
            self.module.fail_json(changed=False, msg='Updating configuration is only supported for LXC enabled proxmox clusters.')
        minimum_version = {'tags': '6.1', 'timezone': '6.3'}
        proxmox_node = self.proxmox_api.nodes(node)
        pve_version = self.version()
        for option, version in minimum_version.items():
            if pve_version < LooseVersion(version) and option in kwargs:
                self.module.fail_json(changed=False, msg="Feature {option} is only supported in PVE {version}+, and you're using PVE {pve_version}".format(option=option, version=version, pve_version=pve_version))
        kwargs = dict(((k, v) for k, v in kwargs.items() if v is not None))
        if cpus is not None:
            kwargs['cpulimit'] = cpus
        if disk is not None:
            kwargs['rootfs'] = disk
        if memory is not None:
            kwargs['memory'] = memory
        if swap is not None:
            kwargs['swap'] = swap
        if 'netif' in kwargs:
            kwargs.update(kwargs['netif'])
            del kwargs['netif']
        if 'mounts' in kwargs:
            kwargs.update(kwargs['mounts'])
            del kwargs['mounts']
        if 'tags' in kwargs:
            re_tag = re.compile('^[a-z0-9_][a-z0-9_\\-\\+\\.]*$')
            for tag in kwargs['tags']:
                if not re_tag.match(tag):
                    self.module.fail_json(msg='%s is not a valid tag' % tag)
            kwargs['tags'] = ','.join(kwargs['tags'])
        current_config = getattr(proxmox_node, VZ_TYPE)(vmid).config.get()
        update_config = False
        for arg, value in kwargs.items():
            if arg not in current_config:
                update_config = True
                break
            if isinstance(value, str):
                current_values = current_config[arg].split(',')
                requested_values = value.split(',')
                for new_value in requested_values:
                    if new_value not in current_values:
                        update_config = True
                        break
            elif str(value) != str(current_config[arg]):
                update_config = True
                break
        if update_config:
            getattr(proxmox_node, VZ_TYPE)(vmid).config.put(vmid=vmid, node=node, **kwargs)
        else:
            self.module.exit_json(changed=False, msg='Container config is already up to date')

    def create_instance(self, vmid, node, disk, storage, cpus, memory, swap, timeout, clone, **kwargs):
        minimum_version = {'tags': '6.1', 'timezone': '6.3'}
        proxmox_node = self.proxmox_api.nodes(node)
        kwargs = dict(((k, v) for k, v in kwargs.items() if v is not None))
        pve_version = self.version()
        for option, version in minimum_version.items():
            if pve_version < LooseVersion(version) and option in kwargs:
                self.module.fail_json(changed=False, msg="Feature {option} is only supported in PVE {version}+, and you're using PVE {pve_version}".format(option=option, version=version, pve_version=pve_version))
        if VZ_TYPE == 'lxc':
            kwargs['cpulimit'] = cpus
            kwargs['rootfs'] = disk
            if 'netif' in kwargs:
                kwargs.update(kwargs['netif'])
                del kwargs['netif']
            if 'mounts' in kwargs:
                kwargs.update(kwargs['mounts'])
                del kwargs['mounts']
            if 'pubkey' in kwargs:
                if self.version() >= LooseVersion('4.2'):
                    kwargs['ssh-public-keys'] = kwargs['pubkey']
                del kwargs['pubkey']
        else:
            kwargs['cpus'] = cpus
            kwargs['disk'] = disk
        if 'tags' in kwargs:
            re_tag = re.compile('^[a-z0-9_][a-z0-9_\\-\\+\\.]*$')
            for tag in kwargs['tags']:
                if not re_tag.match(tag):
                    self.module.fail_json(msg='%s is not a valid tag' % tag)
            kwargs['tags'] = ','.join(kwargs['tags'])
        if kwargs.get('ostype') == 'auto':
            kwargs.pop('ostype')
        if clone is not None:
            if VZ_TYPE != 'lxc':
                self.module.fail_json(changed=False, msg='Clone operator is only supported for LXC enabled proxmox clusters.')
            clone_is_template = self.is_template_container(node, clone)
            create_full_copy = not clone_is_template
            valid_clone_parameters = ['hostname', 'pool', 'description']
            if self.module.params['storage'] is not None and clone_is_template:
                create_full_copy = True
            elif self.module.params['storage'] is None and (not clone_is_template):
                self.module.fail_json(changed=False, msg='Cloned container is not a template, storage needs to be specified.')
            if self.module.params['clone_type'] == 'linked':
                if not clone_is_template:
                    self.module.fail_json(changed=False, msg="'linked' clone type is specified, but cloned container is not a template container.")
            elif self.module.params['clone_type'] == 'opportunistic':
                if not clone_is_template:
                    valid_clone_parameters.append('storage')
            elif self.module.params['clone_type'] == 'full':
                create_full_copy = True
                valid_clone_parameters.append('storage')
            clone_parameters = {}
            if create_full_copy:
                clone_parameters['full'] = '1'
            else:
                clone_parameters['full'] = '0'
            for param in valid_clone_parameters:
                if self.module.params[param] is not None:
                    clone_parameters[param] = self.module.params[param]
            taskid = getattr(proxmox_node, VZ_TYPE)(clone).clone.post(newid=vmid, **clone_parameters)
        else:
            taskid = getattr(proxmox_node, VZ_TYPE).create(vmid=vmid, storage=storage, memory=memory, swap=swap, **kwargs)
        while timeout:
            if self.api_task_ok(node, taskid):
                return True
            timeout -= 1
            if timeout == 0:
                self.module.fail_json(vmid=vmid, node=node, msg='Reached timeout while waiting for creating VM. Last line in task before timeout: %s' % proxmox_node.tasks(taskid).log.get()[:1])
            time.sleep(1)
        return False

    def start_instance(self, vm, vmid, timeout):
        taskid = getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).status.start.post()
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                return True
            timeout -= 1
            if timeout == 0:
                self.module.fail_json(vmid=vmid, taskid=taskid, msg='Reached timeout while waiting for starting VM. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
        return False

    def stop_instance(self, vm, vmid, timeout, force):
        if force:
            taskid = getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).status.shutdown.post(forceStop=1)
        else:
            taskid = getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).status.shutdown.post()
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                return True
            timeout -= 1
            if timeout == 0:
                self.module.fail_json(vmid=vmid, taskid=taskid, msg='Reached timeout while waiting for stopping VM. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
        return False

    def convert_to_template(self, vm, vmid, timeout, force):
        if getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).status.current.get()['status'] == 'running' and force:
            self.stop_instance(vm, vmid, timeout, force)
        getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).template.post()
        return True

    def umount_instance(self, vm, vmid, timeout):
        taskid = getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).status.umount.post()
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                return True
            timeout -= 1
            if timeout == 0:
                self.module.fail_json(vmid=vmid, taskid=taskid, msg='Reached timeout while waiting for unmounting VM. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
        return False