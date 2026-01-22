from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VmwareHostDiskManager(PyVmomi):

    def __init__(self, module):
        super(VmwareHostDiskManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name')
        esxi_host_name = self.params.get('esxi_hostname')
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system with given configuration.')

    def gather_disk_info(self):
        """
        Gather information about SCSI disks

        """
        results = dict(changed=False, hosts_scsidisk_info=dict())
        for host in self.hosts:
            disk_info = []
            storage_system = host.configManager.storageSystem
            for disk in storage_system.storageDeviceInfo.scsiLun:
                temp_disk_info = {'device_name': disk.deviceName, 'device_type': disk.deviceType, 'key': disk.key, 'uuid': disk.uuid, 'canonical_name': disk.canonicalName, 'display_name': disk.displayName, 'lun_type': disk.lunType, 'vendor': disk.vendor, 'model': disk.model, 'revision': disk.revision, 'scsi_level': disk.scsiLevel, 'serial_number': disk.serialNumber, 'vStorageSupport': disk.vStorageSupport, 'protocol_endpoint': disk.protocolEndpoint, 'perenniallyReserved': disk.perenniallyReserved, 'block_size': None, 'block': None, 'device_path': '', 'ssd': False, 'local_disk': False, 'scsi_disk_type': None}
                if hasattr(disk, 'capacity'):
                    temp_disk_info['block_size'] = disk.capacity.blockSize
                    temp_disk_info['block'] = disk.capacity.block
                if hasattr(disk, 'devicePath'):
                    temp_disk_info['device_path'] = disk.devicePath
                if hasattr(disk, 'ssd'):
                    temp_disk_info['ssd'] = disk.ssd
                if hasattr(disk, 'localDisk'):
                    temp_disk_info['local_disk'] = disk.localDisk
                if hasattr(disk, 'scsiDiskType'):
                    temp_disk_info['scsi_disk_type'] = disk.scsiDiskType
                disk_info.append(temp_disk_info)
            results['hosts_scsidisk_info'][host.name] = disk_info
        self.module.exit_json(**results)