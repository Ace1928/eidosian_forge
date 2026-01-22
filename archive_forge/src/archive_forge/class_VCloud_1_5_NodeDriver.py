import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class VCloud_1_5_NodeDriver(VCloudNodeDriver):
    connectionCls = VCloud_1_5_Connection
    NODE_STATE_MAP = {'-1': NodeState.UNKNOWN, '0': NodeState.PENDING, '1': NodeState.PENDING, '2': NodeState.PENDING, '3': NodeState.PENDING, '4': NodeState.RUNNING, '5': NodeState.RUNNING, '6': NodeState.UNKNOWN, '7': NodeState.UNKNOWN, '8': NodeState.STOPPED, '9': NodeState.UNKNOWN, '10': NodeState.UNKNOWN}

    def list_locations(self):
        return [NodeLocation(id=self.connection.host, name=self.connection.host, country='N/A', driver=self)]

    def ex_find_node(self, node_name, vdcs=None):
        """
        Searches for node across specified vDCs. This is more effective than
        querying all nodes to get a single instance.

        :param node_name: The name of the node to search for
        :type node_name: ``str``

        :param vdcs: None, vDC or a list of vDCs to search in. If None all vDCs
                     will be searched.
        :type vdcs: :class:`Vdc`

        :return: node instance or None if not found
        :rtype: :class:`Node` or ``None``
        """
        if not vdcs:
            vdcs = self.vdcs
        if not getattr(vdcs, '__iter__', False):
            vdcs = [vdcs]
        for vdc in vdcs:
            res = self.connection.request(get_url_path(vdc.id))
            xpath = fixxpath(res.object, 'ResourceEntities/ResourceEntity')
            entity_elems = res.object.findall(xpath)
            for entity_elem in entity_elems:
                if entity_elem.get('type') == 'application/vnd.vmware.vcloud.vApp+xml' and entity_elem.get('name') == node_name:
                    path = entity_elem.get('href')
                    return self._ex_get_node(path)
        return None

    def ex_find_vm_nodes(self, vm_name, max_results=50):
        """
        Finds nodes that contain a VM with the specified name.

        :param vm_name: The VM name to find nodes for
        :type vm_name: ``str``

        :param max_results: Maximum number of results up to 128
        :type max_results: ``int``

        :return: List of node instances
        :rtype: `list` of :class:`Node`
        """
        vms = self.ex_query('vm', filter='name=={vm_name}'.format(vm_name=vm_name), page=1, page_size=max_results)
        return [self._ex_get_node(vm['container']) for vm in vms]

    def destroy_node(self, node, shutdown=True):
        try:
            self.ex_undeploy_node(node, shutdown=shutdown)
        except Exception:
            pass
        res = self.connection.request(get_url_path(node.id), method='DELETE')
        return res.status == httplib.ACCEPTED

    def reboot_node(self, node):
        res = self.connection.request('%s/power/action/reset' % get_url_path(node.id), method='POST')
        if res.status in [httplib.ACCEPTED, httplib.NO_CONTENT]:
            self._wait_for_task_completion(res.object.get('href'))
            return True
        else:
            return False

    def ex_deploy_node(self, node, ex_force_customization=False):
        """
        Deploys existing node. Equal to vApp "start" operation.

        :param  node: The node to be deployed
        :type   node: :class:`Node`

        :param  ex_force_customization: Used to specify whether to force
                                        customization on deployment,
                                        if not set default value is False.
        :type   ex_force_customization: ``bool``

        :rtype: :class:`Node`
        """
        if ex_force_customization:
            vms = self._get_vm_elements(node.id)
            for vm in vms:
                self._ex_deploy_node_or_vm(vm.get('href'), ex_force_customization=True)
        else:
            self._ex_deploy_node_or_vm(node.id)
        res = self.connection.request(get_url_path(node.id))
        return self._to_node(res.object)

    def _ex_deploy_node_or_vm(self, vapp_or_vm_path, ex_force_customization=False):
        data = {'powerOn': 'true', 'forceCustomization': str(ex_force_customization).lower(), 'xmlns': 'http://www.vmware.com/vcloud/v1.5'}
        deploy_xml = ET.Element('DeployVAppParams', data)
        path = get_url_path(vapp_or_vm_path)
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.deployVAppParams+xml'}
        res = self.connection.request('%s/action/deploy' % path, data=ET.tostring(deploy_xml), method='POST', headers=headers)
        self._wait_for_task_completion(res.object.get('href'))

    def ex_undeploy_node(self, node, shutdown=True):
        """
        Undeploys existing node. Equal to vApp "stop" operation.

        :param  node: The node to be deployed
        :type   node: :class:`Node`

        :param  shutdown: Whether to shutdown or power off the guest when
                undeploying
        :type   shutdown: ``bool``

        :rtype: :class:`Node`
        """
        data = {'xmlns': 'http://www.vmware.com/vcloud/v1.5'}
        undeploy_xml = ET.Element('UndeployVAppParams', data)
        undeploy_power_action_xml = ET.SubElement(undeploy_xml, 'UndeployPowerAction')
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.undeployVAppParams+xml'}

        def undeploy(action):
            undeploy_power_action_xml.text = action
            undeploy_res = self.connection.request('%s/action/undeploy' % get_url_path(node.id), data=ET.tostring(undeploy_xml), method='POST', headers=headers)
            self._wait_for_task_completion(undeploy_res.object.get('href'))
        if shutdown:
            try:
                undeploy('shutdown')
            except Exception:
                undeploy('powerOff')
        else:
            undeploy('powerOff')
        res = self.connection.request(get_url_path(node.id))
        return self._to_node(res.object)

    def ex_power_off_node(self, node):
        """
        Powers on all VMs under specified node. VMs need to be This operation
        is allowed only when the vApp/VM is powered on.

        :param  node: The node to be powered off
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
        return self._perform_power_operation(node, 'powerOff')

    def ex_power_on_node(self, node):
        """
        Powers on all VMs under specified node. This operation is allowed
        only when the vApp/VM is powered off or suspended.

        :param  node: The node to be powered on
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
        return self._perform_power_operation(node, 'powerOn')

    def ex_shutdown_node(self, node):
        """
        Shutdowns all VMs under specified node. This operation is allowed only
        when the vApp/VM is powered on.

        :param  node: The node to be shut down
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
        return self._perform_power_operation(node, 'shutdown')

    def ex_suspend_node(self, node):
        """
        Suspends all VMs under specified node. This operation is allowed only
        when the vApp/VM is powered on.

        :param  node: The node to be suspended
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
        return self._perform_power_operation(node, 'suspend')

    def _perform_power_operation(self, node, operation):
        res = self.connection.request('{}/power/action/{}'.format(get_url_path(node.id), operation), method='POST')
        self._wait_for_task_completion(res.object.get('href'))
        res = self.connection.request(get_url_path(node.id))
        return self._to_node(res.object)

    def ex_get_control_access(self, node):
        """
        Returns the control access settings for specified node.

        :param  node: node to get the control access for
        :type   node: :class:`Node`

        :rtype: :class:`ControlAccess`
        """
        res = self.connection.request('%s/controlAccess' % get_url_path(node.id))
        everyone_access_level = None
        is_shared_elem = res.object.find(fixxpath(res.object, 'IsSharedToEveryone'))
        if is_shared_elem is not None and is_shared_elem.text == 'true':
            everyone_access_level = res.object.find(fixxpath(res.object, 'EveryoneAccessLevel')).text
        subjects = []
        xpath = fixxpath(res.object, 'AccessSettings/AccessSetting')
        for elem in res.object.findall(xpath):
            access_level = elem.find(fixxpath(res.object, 'AccessLevel')).text
            subject_elem = elem.find(fixxpath(res.object, 'Subject'))
            if subject_elem.get('type') == 'application/vnd.vmware.admin.group+xml':
                subj_type = 'group'
            else:
                subj_type = 'user'
            path = get_url_path(subject_elem.get('href'))
            res = self.connection.request(path)
            name = res.object.get('name')
            subject = Subject(type=subj_type, name=name, access_level=access_level, id=subject_elem.get('href'))
            subjects.append(subject)
        return ControlAccess(node, everyone_access_level, subjects)

    def ex_set_control_access(self, node, control_access):
        """
        Sets control access for the specified node.

        :param  node: node
        :type   node: :class:`Node`

        :param  control_access: control access settings
        :type   control_access: :class:`ControlAccess`

        :rtype: ``None``
        """
        xml = ET.Element('ControlAccessParams', {'xmlns': 'http://www.vmware.com/vcloud/v1.5'})
        shared_to_everyone = ET.SubElement(xml, 'IsSharedToEveryone')
        if control_access.everyone_access_level:
            shared_to_everyone.text = 'true'
            everyone_access_level = ET.SubElement(xml, 'EveryoneAccessLevel')
            everyone_access_level.text = control_access.everyone_access_level
        else:
            shared_to_everyone.text = 'false'
        if control_access.subjects:
            access_settings_elem = ET.SubElement(xml, 'AccessSettings')
        for subject in control_access.subjects:
            setting = ET.SubElement(access_settings_elem, 'AccessSetting')
            if subject.id:
                href = subject.id
            else:
                res = self.ex_query(type=subject.type, filter='name==' + subject.name)
                if not res:
                    raise LibcloudError('Specified subject "{} {}" not found '.format(subject.type, subject.name))
                href = res[0]['href']
            ET.SubElement(setting, 'Subject', {'href': href})
            ET.SubElement(setting, 'AccessLevel').text = subject.access_level
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.controlAccess+xml'}
        self.connection.request('%s/action/controlAccess' % get_url_path(node.id), data=ET.tostring(xml), headers=headers, method='POST')

    def ex_get_metadata(self, node):
        """
        :param  node: node
        :type   node: :class:`Node`

        :return: dictionary mapping metadata keys to metadata values
        :rtype: dictionary mapping ``str`` to ``str``
        """
        res = self.connection.request('%s/metadata' % get_url_path(node.id))
        xpath = fixxpath(res.object, 'MetadataEntry')
        metadata_entries = res.object.findall(xpath)
        res_dict = {}
        for entry in metadata_entries:
            key = entry.findtext(fixxpath(res.object, 'Key'))
            value = entry.findtext(fixxpath(res.object, 'Value'))
            res_dict[key] = value
        return res_dict

    def ex_set_metadata_entry(self, node, key, value):
        """
        :param  node: node
        :type   node: :class:`Node`

        :param key: metadata key to be set
        :type key: ``str``

        :param value: metadata value to be set
        :type value: ``str``

        :rtype: ``None``
        """
        metadata_elem = ET.Element('Metadata', {'xmlns': 'http://www.vmware.com/vcloud/v1.5', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
        entry = ET.SubElement(metadata_elem, 'MetadataEntry')
        key_elem = ET.SubElement(entry, 'Key')
        key_elem.text = key
        value_elem = ET.SubElement(entry, 'Value')
        value_elem.text = value
        res = self.connection.request('%s/metadata' % get_url_path(node.id), data=ET.tostring(metadata_elem), headers={'Content-Type': 'application/vnd.vmware.vcloud.metadata+xml'}, method='POST')
        self._wait_for_task_completion(res.object.get('href'))

    def ex_query(self, type, filter=None, page=1, page_size=100, sort_asc=None, sort_desc=None):
        """
        Queries vCloud for specified type. See
        http://www.vmware.com/pdf/vcd_15_api_guide.pdf for details. Each
        element of the returned list is a dictionary with all attributes from
        the record.

        :param type: type to query (r.g. user, group, vApp etc.)
        :type  type: ``str``

        :param filter: filter expression (see documentation for syntax)
        :type  filter: ``str``

        :param page: page number
        :type  page: ``int``

        :param page_size: page size
        :type  page_size: ``int``

        :param sort_asc: sort in ascending order by specified field
        :type  sort_asc: ``str``

        :param sort_desc: sort in descending order by specified field
        :type  sort_desc: ``str``

        :rtype: ``list`` of dict
        """
        params = {'type': type, 'pageSize': page_size, 'page': page}
        if sort_asc:
            params['sortAsc'] = sort_asc
        if sort_desc:
            params['sortDesc'] = sort_desc
        url = '/api/query?' + urlencode(params)
        if filter:
            if not filter.startswith('('):
                filter = '(' + filter + ')'
            url += '&filter=' + filter.replace(' ', '+')
        results = []
        res = self.connection.request(url)
        for elem in res.object:
            if not elem.tag.endswith('Link'):
                result = elem.attrib
                result['type'] = elem.tag.split('}')[1]
                results.append(result)
        return results

    def create_node(self, **kwargs):
        """
        Creates and returns node. If the source image is:
          - vApp template - a new vApp is instantiated from template
          - existing vApp - a new vApp is cloned from the source vApp. Can
                            not clone more vApps is parallel otherwise
                            resource busy error is raised.


        @inherits: :class:`NodeDriver.create_node`

        :keyword    image:  OS Image to boot on node. (required). Can be a
                            NodeImage or existing Node that will be cloned.
        :type       image:  :class:`NodeImage` or :class:`Node`

        :keyword    ex_network: Organisation's network name for attaching vApp
                                VMs to.
        :type       ex_network: ``str``

        :keyword    ex_vdc: Name of organisation's virtual data center where
                            vApp VMs will be deployed.
        :type       ex_vdc: ``str``

        :keyword    ex_vm_names: list of names to be used as a VM and computer
                                 name. The name must be max. 15 characters
                                 long and follow the host name requirements.
        :type       ex_vm_names: ``list`` of ``str``

        :keyword    ex_vm_cpu: number of virtual CPUs/cores to allocate for
                               each vApp VM.
        :type       ex_vm_cpu: ``int``

        :keyword    ex_vm_memory: amount of memory in MB to allocate for each
                                  vApp VM.
        :type       ex_vm_memory: ``int``

        :keyword    ex_vm_script: full path to file containing guest
                                  customisation script for each vApp VM.
                                  Useful for creating users & pushing out
                                  public SSH keys etc.
        :type       ex_vm_script: ``str``

        :keyword    ex_vm_script_text: content of guest customisation script
                                       for each vApp VM. Overrides ex_vm_script
                                       parameter.
        :type       ex_vm_script_text: ``str``

        :keyword    ex_vm_network: Override default vApp VM network name.
                                   Useful for when you've imported an OVF
                                   originating from outside of the vCloud.
        :type       ex_vm_network: ``str``

        :keyword    ex_vm_fence: Fence mode for connecting the vApp VM network
                                 (ex_vm_network) to the parent
                                 organisation network (ex_network).
        :type       ex_vm_fence: ``str``

        :keyword    ex_vm_ipmode: IP address allocation mode for all vApp VM
                                  network connections.
        :type       ex_vm_ipmode: ``str``

        :keyword    ex_deploy: set to False if the node shouldn't be deployed
                               (started) after creation
        :type       ex_deploy: ``bool``

        :keyword    ex_force_customization: Used to specify whether to force
                                            customization on deployment,
                                            if not set default value is False.
        :type       ex_force_customization: ``bool``

        :keyword    ex_clone_timeout: timeout in seconds for clone/instantiate
                                      VM operation.
                                      Cloning might be a time consuming
                                      operation especially when linked clones
                                      are disabled or VMs are created on
                                      different datastores.
                                      Overrides the default task completion
                                      value.
        :type       ex_clone_timeout: ``int``

        :keyword    ex_admin_password: set the node admin password explicitly.
        :type       ex_admin_password: ``str``

        :keyword    ex_description: Set a description for the vApp.
        :type       ex_description: ``str``
        """
        name = kwargs['name']
        image = kwargs['image']
        ex_vm_names = kwargs.get('ex_vm_names')
        ex_vm_cpu = kwargs.get('ex_vm_cpu')
        ex_vm_memory = kwargs.get('ex_vm_memory')
        ex_vm_script = kwargs.get('ex_vm_script')
        ex_vm_script_text = kwargs.get('ex_vm_script_text', None)
        ex_vm_fence = kwargs.get('ex_vm_fence', None)
        ex_network = kwargs.get('ex_network', None)
        ex_vm_network = kwargs.get('ex_vm_network', None)
        ex_vm_ipmode = kwargs.get('ex_vm_ipmode', None)
        ex_deploy = kwargs.get('ex_deploy', True)
        ex_force_customization = kwargs.get('ex_force_customization', False)
        ex_vdc = kwargs.get('ex_vdc', None)
        ex_clone_timeout = kwargs.get('ex_clone_timeout', DEFAULT_TASK_COMPLETION_TIMEOUT)
        ex_admin_password = kwargs.get('ex_admin_password', None)
        ex_description = kwargs.get('ex_description', None)
        self._validate_vm_names(ex_vm_names)
        self._validate_vm_cpu(ex_vm_cpu)
        self._validate_vm_memory(ex_vm_memory)
        self._validate_vm_fence(ex_vm_fence)
        self._validate_vm_ipmode(ex_vm_ipmode)
        ex_vm_script = self._validate_vm_script(ex_vm_script)
        if ex_network:
            network_href = self._get_network_href(ex_network)
            network_elem = self.connection.request(get_url_path(network_href)).object
        else:
            network_elem = None
        vdc = self._get_vdc(ex_vdc)
        if self._is_node(image):
            vapp_name, vapp_href = self._clone_node(name, image, vdc, ex_clone_timeout)
        else:
            vapp_name, vapp_href = self._instantiate_node(name, image, network_elem, vdc, ex_vm_network, ex_vm_fence, ex_clone_timeout, description=ex_description)
        self._change_vm_names(vapp_href, ex_vm_names)
        self._change_vm_cpu(vapp_href, ex_vm_cpu)
        self._change_vm_memory(vapp_href, ex_vm_memory)
        self._change_vm_script(vapp_href, ex_vm_script, ex_vm_script_text)
        self._change_vm_ipmode(vapp_href, ex_vm_ipmode)
        if ex_admin_password is not None:
            self.ex_change_vm_admin_password(vapp_href, ex_admin_password)
        if ex_deploy:
            res = self.connection.request(get_url_path(vapp_href))
            node = self._to_node(res.object)
            retry = 3
            while True:
                try:
                    self.ex_deploy_node(node, ex_force_customization)
                    break
                except Exception:
                    if retry <= 0:
                        raise
                    retry -= 1
                    time.sleep(10)
        res = self.connection.request(get_url_path(vapp_href))
        node = self._to_node(res.object)
        return node

    def _instantiate_node(self, name, image, network_elem, vdc, vm_network, vm_fence, instantiate_timeout, description=None):
        instantiate_xml = Instantiate_1_5_VAppXML(name=name, template=image.id, network=network_elem, vm_network=vm_network, vm_fence=vm_fence, description=description)
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.instantiateVAppTemplateParams+xml'}
        res = self.connection.request('%s/action/instantiateVAppTemplate' % get_url_path(vdc.id), data=instantiate_xml.tostring(), method='POST', headers=headers)
        vapp_name = res.object.get('name')
        vapp_href = res.object.get('href')
        task_href = res.object.find(fixxpath(res.object, 'Tasks/Task')).get('href')
        self._wait_for_task_completion(task_href, instantiate_timeout)
        return (vapp_name, vapp_href)

    def _clone_node(self, name, sourceNode, vdc, clone_timeout):
        clone_xml = ET.Element('CloneVAppParams', {'name': name, 'deploy': 'false', 'powerOn': 'false', 'xmlns': 'http://www.vmware.com/vcloud/v1.5', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
        ET.SubElement(clone_xml, 'Description').text = 'Clone of ' + sourceNode.name
        ET.SubElement(clone_xml, 'Source', {'href': sourceNode.id})
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.cloneVAppParams+xml'}
        res = self.connection.request('%s/action/cloneVApp' % get_url_path(vdc.id), data=ET.tostring(clone_xml), method='POST', headers=headers)
        vapp_name = res.object.get('name')
        vapp_href = res.object.get('href')
        task_href = res.object.find(fixxpath(res.object, 'Tasks/Task')).get('href')
        self._wait_for_task_completion(task_href, clone_timeout)
        res = self.connection.request(get_url_path(vapp_href))
        vms = res.object.findall(fixxpath(res.object, 'Children/Vm'))
        for i, vm in enumerate(vms):
            network_xml = ET.Element('NetworkConnectionSection', {'ovf:required': 'false', 'xmlns': 'http://www.vmware.com/vcloud/v1.5', 'xmlns:ovf': 'http://schemas.dmtf.org/ovf/envelope/1'})
            ET.SubElement(network_xml, 'ovf:Info').text = 'Specifies the available VM network connections'
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.networkConnectionSection+xml'}
            res = self.connection.request('%s/networkConnectionSection' % get_url_path(vm.get('href')), data=ET.tostring(network_xml), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))
            network_xml = vm.find(fixxpath(vm, 'NetworkConnectionSection'))
            network_conn_xml = network_xml.find(fixxpath(network_xml, 'NetworkConnection'))
            network_conn_xml.set('needsCustomization', 'true')
            network_conn_xml.remove(network_conn_xml.find(fixxpath(network_xml, 'IpAddress')))
            network_conn_xml.remove(network_conn_xml.find(fixxpath(network_xml, 'MACAddress')))
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.networkConnectionSection+xml'}
            res = self.connection.request('%s/networkConnectionSection' % get_url_path(vm.get('href')), data=ET.tostring(network_xml), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))
        return (vapp_name, vapp_href)

    def ex_set_vm_cpu(self, vapp_or_vm_id, vm_cpu):
        """
        Sets the number of virtual CPUs for the specified VM or VMs under
        the vApp. If the vapp_or_vm_id param represents a link to an vApp
        all VMs that are attached to this vApp will be modified.

        Please ensure that hot-adding a virtual CPU is enabled for the
        powered on virtual machines. Otherwise use this method on undeployed
        vApp.

        :keyword    vapp_or_vm_id: vApp or VM ID that will be modified. If
                                   a vApp ID is used here all attached VMs
                                   will be modified
        :type       vapp_or_vm_id: ``str``

        :keyword    vm_cpu: number of virtual CPUs/cores to allocate for
                            specified VMs
        :type       vm_cpu: ``int``

        :rtype: ``None``
        """
        self._validate_vm_cpu(vm_cpu)
        self._change_vm_cpu(vapp_or_vm_id, vm_cpu)

    def ex_set_vm_memory(self, vapp_or_vm_id, vm_memory):
        """
        Sets the virtual memory in MB to allocate for the specified VM or
        VMs under the vApp. If the vapp_or_vm_id param represents a link
        to an vApp all VMs that are attached to this vApp will be modified.

        Please ensure that hot-change of virtual memory is enabled for the
        powered on virtual machines. Otherwise use this method on undeployed
        vApp.

        :keyword    vapp_or_vm_id: vApp or VM ID that will be modified. If
                                   a vApp ID is used here all attached VMs
                                   will be modified
        :type       vapp_or_vm_id: ``str``

        :keyword    vm_memory: virtual memory in MB to allocate for the
                               specified VM or VMs
        :type       vm_memory: ``int``

        :rtype: ``None``
        """
        self._validate_vm_memory(vm_memory)
        self._change_vm_memory(vapp_or_vm_id, vm_memory)

    def ex_add_vm_disk(self, vapp_or_vm_id, vm_disk_size):
        """
        Adds a virtual disk to the specified VM or VMs under the vApp. If the
        vapp_or_vm_id param represents a link to an vApp all VMs that are
        attached to this vApp will be modified.

        :keyword    vapp_or_vm_id: vApp or VM ID that will be modified. If a
                                   vApp ID is used here all attached VMs
                                   will be modified
        :type       vapp_or_vm_id: ``str``

        :keyword    vm_disk_size: the disk capacity in GB that will be added
                                  to the specified VM or VMs
        :type       vm_disk_size: ``int``

        :rtype: ``None``
        """
        self._validate_vm_disk_size(vm_disk_size)
        self._add_vm_disk(vapp_or_vm_id, vm_disk_size)

    @staticmethod
    def _validate_vm_names(names):
        if names is None:
            return
        hname_re = re.compile('^(([a-zA-Z]|[a-zA-Z][a-zA-Z0-9]*)[\\-])*([A-Za-z]|[A-Za-z][A-Za-z0-9]*[A-Za-z0-9])$')
        for name in names:
            if len(name) > 15:
                raise ValueError('The VM name "' + name + '" is too long for the computer name (max 15 chars allowed).')
            if not hname_re.match(name):
                raise ValueError('The VM name "' + name + '" can not be used. "' + name + '" is not a valid computer name for the VM.')

    @staticmethod
    def _validate_vm_memory(vm_memory):
        if vm_memory is None:
            return
        elif vm_memory not in VIRTUAL_MEMORY_VALS:
            raise ValueError('%s is not a valid vApp VM memory value' % vm_memory)

    @staticmethod
    def _validate_vm_cpu(vm_cpu):
        if vm_cpu is None:
            return
        elif vm_cpu not in VIRTUAL_CPU_VALS_1_5:
            raise ValueError('%s is not a valid vApp VM CPU value' % vm_cpu)

    @staticmethod
    def _validate_vm_disk_size(vm_disk):
        if vm_disk is None:
            return
        elif int(vm_disk) < 0:
            raise ValueError('%s is not a valid vApp VM disk space value', vm_disk)

    @staticmethod
    def _validate_vm_script(vm_script):
        if vm_script is None:
            return
        if not os.path.isabs(vm_script):
            vm_script = os.path.expanduser(vm_script)
            vm_script = os.path.abspath(vm_script)
        if not os.path.isfile(vm_script):
            raise LibcloudError('%s the VM script file does not exist' % vm_script)
        try:
            open(vm_script).read()
        except Exception:
            raise
        return vm_script

    @staticmethod
    def _validate_vm_fence(vm_fence):
        if vm_fence is None:
            return
        elif vm_fence not in FENCE_MODE_VALS_1_5:
            raise ValueError('%s is not a valid fencing mode value' % vm_fence)

    @staticmethod
    def _validate_vm_ipmode(vm_ipmode):
        if vm_ipmode is None:
            return
        elif vm_ipmode == 'MANUAL':
            raise NotImplementedError('MANUAL IP mode: The interface for supplying IPAddress does not exist yet')
        elif vm_ipmode not in IP_MODE_VALS_1_5:
            raise ValueError('%s is not a valid IP address allocation mode value' % vm_ipmode)

    def _change_vm_names(self, vapp_or_vm_id, vm_names):
        if vm_names is None:
            return
        vms = self._get_vm_elements(vapp_or_vm_id)
        for i, vm in enumerate(vms):
            if len(vm_names) <= i:
                return
            res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')))
            res.object.find(fixxpath(res.object, 'ComputerName')).text = vm_names[i]
            self._remove_admin_password(res.object)
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.guestCustomizationSection+xml'}
            res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))
            req_xml = ET.Element('Vm', {'name': vm_names[i], 'xmlns': 'http://www.vmware.com/vcloud/v1.5'})
            res = self.connection.request(get_url_path(vm.get('href')), data=ET.tostring(req_xml), method='PUT', headers={'Content-Type': 'application/vnd.vmware.vcloud.vm+xml'})
            self._wait_for_task_completion(res.object.get('href'))

    def _change_vm_cpu(self, vapp_or_vm_id, vm_cpu):
        if vm_cpu is None:
            return
        vms = self._get_vm_elements(vapp_or_vm_id)
        for vm in vms:
            res = self.connection.request('%s/virtualHardwareSection/cpu' % get_url_path(vm.get('href')))
            xpath = '{http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData}VirtualQuantity'
            res.object.find(xpath).text = str(vm_cpu)
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.rasdItem+xml'}
            res = self.connection.request('%s/virtualHardwareSection/cpu' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))

    def _change_vm_memory(self, vapp_or_vm_id, vm_memory):
        if vm_memory is None:
            return
        vms = self._get_vm_elements(vapp_or_vm_id)
        for vm in vms:
            res = self.connection.request('%s/virtualHardwareSection/memory' % get_url_path(vm.get('href')))
            xpath = '{http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData}VirtualQuantity'
            res.object.find(xpath).text = str(vm_memory)
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.rasdItem+xml'}
            res = self.connection.request('%s/virtualHardwareSection/memory' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))

    def _add_vm_disk(self, vapp_or_vm_id, vm_disk):
        if vm_disk is None:
            return
        rasd_ns = '{http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData}'
        vms = self._get_vm_elements(vapp_or_vm_id)
        for vm in vms:
            res = self.connection.request('%s/virtualHardwareSection/disks' % get_url_path(vm.get('href')))
            existing_ids = []
            new_disk = None
            for item in res.object.findall(fixxpath(res.object, 'Item')):
                for elem in item:
                    if elem.tag == '%sInstanceID' % rasd_ns:
                        existing_ids.append(int(elem.text))
                    if elem.tag in ['%sAddressOnParent' % rasd_ns, '%sParent' % rasd_ns]:
                        item.remove(elem)
                if item.find('%sHostResource' % rasd_ns) is not None:
                    new_disk = item
            new_disk = copy.deepcopy(new_disk)
            disk_id = max(existing_ids) + 1
            new_disk.find('%sInstanceID' % rasd_ns).text = str(disk_id)
            new_disk.find('%sElementName' % rasd_ns).text = 'Hard Disk ' + str(disk_id)
            new_disk.find('%sHostResource' % rasd_ns).set(fixxpath(new_disk, 'capacity'), str(int(vm_disk) * 1024))
            res.object.append(new_disk)
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.rasditemslist+xml'}
            res = self.connection.request('%s/virtualHardwareSection/disks' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))

    def _change_vm_script(self, vapp_or_vm_id, vm_script, vm_script_text=None):
        if vm_script is None and vm_script_text is None:
            return
        if vm_script_text is not None:
            script = vm_script_text
        else:
            try:
                with open(vm_script) as fp:
                    script = fp.read()
            except Exception:
                return
        vms = self._get_vm_elements(vapp_or_vm_id)
        for vm in vms:
            res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')))
            try:
                res.object.find(fixxpath(res.object, 'CustomizationScript')).text = script
            except Exception:
                for i, e in enumerate(res.object):
                    if e.tag == '{http://www.vmware.com/vcloud/v1.5}ComputerName':
                        break
                e = ET.Element('{http://www.vmware.com/vcloud/v1.5}CustomizationScript')
                e.text = script
                res.object.insert(i, e)
            self._remove_admin_password(res.object)
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.guestCustomizationSection+xml'}
            res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))

    def _change_vm_ipmode(self, vapp_or_vm_id, vm_ipmode):
        if vm_ipmode is None:
            return
        vms = self._get_vm_elements(vapp_or_vm_id)
        for vm in vms:
            res = self.connection.request('%s/networkConnectionSection' % get_url_path(vm.get('href')))
            net_conns = res.object.findall(fixxpath(res.object, 'NetworkConnection'))
            for c in net_conns:
                c.find(fixxpath(c, 'IpAddressAllocationMode')).text = vm_ipmode
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.networkConnectionSection+xml'}
            res = self.connection.request('%s/networkConnectionSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))

    @staticmethod
    def _remove_admin_password(guest_customization_section):
        """
        Remove AdminPassword element from GuestCustomizationSection if it
        would be invalid to include it.

        This was originally done unconditionally due to an "API quirk" of
        unknown origin or effect. When AdminPasswordEnabled is set to true
        and AdminPasswordAuto is false, the admin password must be set or
        an error will ensue, and vice versa.
        :param guest_customization_section: GuestCustomizationSection element
                                            to remove password from (if valid
                                            to do so)
        :type guest_customization_section: ``ET.Element``
        """
        admin_pass_enabled = guest_customization_section.find(fixxpath(guest_customization_section, 'AdminPasswordEnabled'))
        admin_pass_auto = guest_customization_section.find(fixxpath(guest_customization_section, 'AdminPasswordAuto'))
        admin_pass = guest_customization_section.find(fixxpath(guest_customization_section, 'AdminPassword'))
        if admin_pass is not None and (admin_pass_enabled is None or admin_pass_enabled.text != 'true' or admin_pass_auto is None or (admin_pass_auto.text != 'false')):
            guest_customization_section.remove(admin_pass)

    def _update_or_insert_section(self, res, section, prev_section, text):
        try:
            res.object.find(fixxpath(res.object, section)).text = text
        except Exception:
            for i, e in enumerate(res.object):
                tag = '{http://www.vmware.com/vcloud/v1.5}%s' % prev_section
                if e.tag == tag:
                    break
            e = ET.Element('{http://www.vmware.com/vcloud/v1.5}%s' % section)
            e.text = text
            res.object.insert(i, e)
        return res

    def ex_change_vm_admin_password(self, vapp_or_vm_id, ex_admin_password):
        """
        Changes the admin (or root) password of VM or VMs under the vApp. If
        the vapp_or_vm_id param represents a link to an vApp all VMs that
        are attached to this vApp will be modified.

        :keyword    vapp_or_vm_id: vApp or VM ID that will be modified. If a
                                   vApp ID is used here all attached VMs
                                   will be modified
        :type       vapp_or_vm_id: ``str``

        :keyword    ex_admin_password: admin password to be used.
        :type       ex_admin_password: ``str``

        :rtype: ``None``
        """
        if ex_admin_password is None:
            return
        vms = self._get_vm_elements(vapp_or_vm_id)
        for vm in vms:
            res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')))
            headers = {'Content-Type': 'application/vnd.vmware.vcloud.guestCustomizationSection+xml'}
            auto_logon = res.object.find(fixxpath(res.object, 'AdminAutoLogonEnabled'))
            if auto_logon is not None and auto_logon.text == 'false':
                self._update_or_insert_section(res, 'AdminAutoLogonCount', 'ResetPasswordRequired', '0')
            self._update_or_insert_section(res, 'AdminPasswordAuto', 'AdminPassword', 'false')
            self._update_or_insert_section(res, 'AdminPasswordEnabled', 'AdminPasswordAuto', 'true')
            self._update_or_insert_section(res, 'AdminPassword', 'AdminAutoLogonEnabled', ex_admin_password)
            res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
            self._wait_for_task_completion(res.object.get('href'))

    def _get_network_href(self, network_name):
        network_href = None
        res = self.connection.request(self.org)
        links = res.object.findall(fixxpath(res.object, 'Link'))
        for link in links:
            if link.attrib['type'] == 'application/vnd.vmware.vcloud.orgNetwork+xml' and link.attrib['name'] == network_name:
                network_href = link.attrib['href']
        if network_href is None:
            raise ValueError('%s is not a valid organisation network name' % network_name)
        else:
            return network_href

    def _ex_get_node(self, node_id):
        """
        Get a node instance from a node ID.

        :param node_id: ID of the node
        :type node_id: ``str``

        :return: node instance or None if not found
        :rtype: :class:`Node` or ``None``
        """
        res = self.connection.request(get_url_path(node_id), headers={'Content-Type': 'application/vnd.vmware.vcloud.vApp+xml'})
        return self._to_node(res.object)

    def _get_vm_elements(self, vapp_or_vm_id):
        res = self.connection.request(get_url_path(vapp_or_vm_id))
        if res.object.tag.endswith('VApp'):
            vms = res.object.findall(fixxpath(res.object, 'Children/Vm'))
        elif res.object.tag.endswith('Vm'):
            vms = [res.object]
        else:
            raise ValueError('Specified ID value is not a valid VApp or Vm identifier.')
        return vms

    def _is_node(self, node_or_image):
        return isinstance(node_or_image, Node)

    def _to_node(self, node_elm):
        if node_elm.find(fixxpath(node_elm, 'SnapshotSection')) is None:
            snapshots = None
        else:
            snapshots = []
            for snapshot_elem in node_elm.findall(fixxpath(node_elm, 'SnapshotSection/Snapshot')):
                snapshots.append({'created': snapshot_elem.get('created'), 'poweredOn': snapshot_elem.get('poweredOn'), 'size': snapshot_elem.get('size')})
        vms = []
        for vm_elem in node_elm.findall(fixxpath(node_elm, 'Children/Vm')):
            public_ips = []
            private_ips = []
            xpath = fixxpath(vm_elem, 'NetworkConnectionSection/NetworkConnection')
            for connection in vm_elem.findall(xpath):
                ip = connection.find(fixxpath(connection, 'IpAddress'))
                if ip is not None:
                    private_ips.append(ip.text)
                external_ip = connection.find(fixxpath(connection, 'ExternalIpAddress'))
                if external_ip is not None:
                    public_ips.append(external_ip.text)
                elif ip is not None:
                    public_ips.append(ip.text)
            xpath = '{http://schemas.dmtf.org/ovf/envelope/1}OperatingSystemSection'
            os_type_elem = vm_elem.find(xpath)
            if os_type_elem is not None:
                os_type = os_type_elem.get('{http://www.vmware.com/schema/ovf}osType')
            else:
                os_type = None
            vm = {'id': vm_elem.get('href'), 'name': vm_elem.get('name'), 'state': self.NODE_STATE_MAP[vm_elem.get('status')], 'public_ips': public_ips, 'private_ips': private_ips, 'os_type': os_type}
            vms.append(vm)
        public_ips = []
        private_ips = []
        for vm in vms:
            public_ips.extend(vm['public_ips'])
            private_ips.extend(vm['private_ips'])
        vdc_id = next((link.get('href') for link in node_elm.findall(fixxpath(node_elm, 'Link')) if link.get('type') == 'application/vnd.vmware.vcloud.vdc+xml'))
        vdc = next((vdc for vdc in self.vdcs if vdc.id == vdc_id))
        extra = {'vdc': vdc.name, 'vms': vms}
        description = node_elm.find(fixxpath(node_elm, 'Description'))
        if description is not None:
            extra['description'] = description.text
        else:
            extra['description'] = ''
        lease_settings = node_elm.find(fixxpath(node_elm, 'LeaseSettingsSection'))
        if lease_settings is not None:
            extra['lease_settings'] = Lease.to_lease(lease_settings)
        else:
            extra['lease_settings'] = None
        if snapshots is not None:
            extra['snapshots'] = snapshots
        node = Node(id=node_elm.get('href'), name=node_elm.get('name'), state=self.NODE_STATE_MAP[node_elm.get('status')], public_ips=public_ips, private_ips=private_ips, driver=self.connection.driver, extra=extra)
        return node

    def _to_vdc(self, vdc_elm):

        def get_capacity_values(capacity_elm):
            if capacity_elm is None:
                return None
            limit = int(capacity_elm.findtext(fixxpath(capacity_elm, 'Limit')))
            used = int(capacity_elm.findtext(fixxpath(capacity_elm, 'Used')))
            units = capacity_elm.findtext(fixxpath(capacity_elm, 'Units'))
            return Capacity(limit, used, units)
        cpu = get_capacity_values(vdc_elm.find(fixxpath(vdc_elm, 'ComputeCapacity/Cpu')))
        memory = get_capacity_values(vdc_elm.find(fixxpath(vdc_elm, 'ComputeCapacity/Memory')))
        storage = get_capacity_values(vdc_elm.find(fixxpath(vdc_elm, 'StorageCapacity')))
        return Vdc(id=vdc_elm.get('href'), name=vdc_elm.get('name'), driver=self, allocation_model=vdc_elm.findtext(fixxpath(vdc_elm, 'AllocationModel')), cpu=cpu, memory=memory, storage=storage)