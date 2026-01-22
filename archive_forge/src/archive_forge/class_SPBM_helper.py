from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
class SPBM_helper(SPBM):

    def __init__(self, module):
        super().__init__(module)

    def SearchStorageProfileByName(self, profileManager, name):
        """
        Search VMware storage policy profile by name.

        :param profileManager: A VMware Storage Policy Service manager object.
        :type profileManager: pbm.profile.ProfileManager
        :param name: A VMware Storage Policy profile name.
        :type name: str
        :returns: A VMware Storage Policy profile object.
        :rtype: pbm.profile.Profile
        """
        profileIds = profileManager.PbmQueryProfile(resourceType=pbm.profile.ResourceType(resourceType='STORAGE'), profileCategory='REQUIREMENT')
        if len(profileIds) > 0:
            storageProfiles = profileManager.PbmRetrieveContent(profileIds=profileIds)
        for storageProfile in storageProfiles:
            if storageProfile.name == name:
                return storageProfile

    def CheckAssociatedStorageProfile(self, profileManager, ref, name):
        """
        Check the associated storage policy profile.

        :param profileManager: A VMware Storage Policy Service manager object.
        :type profileManager: pbm.profile.ProfileManager
        :param ref: A server object ref to a virtual machine, virtual disk,
            or datastore.
        :type ref: pbm.ServerObjectRef
        :param name: A VMware storage policy profile name.
        :type name: str
        :returns: True if storage policy profile by name is associated to ref.
        :rtype: bool
        """
        profileIds = profileManager.PbmQueryAssociatedProfile(ref)
        if len(profileIds) > 0:
            profiles = profileManager.PbmRetrieveContent(profileIds=profileIds)
            for profile in profiles:
                if profile.name == name:
                    return True
        return False

    def SetVMHomeStorageProfile(self, vm, profile):
        """
        Set VM Home storage policy profile.

        :param vm: A virtual machine object.
        :type vm: VirtualMachine
        :param profile: A VMware Storage Policy profile.
        :type profile: pbm.profile.Profile
        :returns: VMware task object.
        :rtype: Task
        """
        spec = vim.vm.ConfigSpec()
        profileSpec = vim.vm.DefinedProfileSpec()
        profileSpec.profileId = profile.profileId.uniqueId
        spec.vmProfile = [profileSpec]
        return vm.ReconfigVM_Task(spec)

    def GetVirtualDiskObj(self, vm, unit_number, controller_number):
        """
        Get a virtual disk object.

        :param vm: A virtual machine object.
        :type vm: VirtualMachine
        :param unit_number: virtual machine's disk unit number.
        :type unit_number: int
        :param controller_number: virtual machine's controller number.
        :type controller_number: int
        :returns: VirtualDisk object if exists, else None.
        :rtype: VirtualDisk, None
        """
        controllerKey = None
        for device in vm.config.hardware.device:
            if isinstance(device, vim.vm.device.VirtualSCSIController):
                if device.busNumber == controller_number:
                    controllerKey = device.key
                    break
        if controllerKey is not None:
            for device in vm.config.hardware.device:
                if not isinstance(device, vim.vm.device.VirtualDisk):
                    continue
                if int(device.unitNumber) == int(unit_number) and int(device.controllerKey) == controllerKey:
                    return device
        return None

    def SetVMDiskStorageProfile(self, vm, unit_number, controller_number, profile):
        """
        Set VM's disk storage policy profile.

        :param vm: A virtual machine object
        :type vm: VirtualMachine
        :param unit_number: virtual machine's disk unit number.
        :type unit_number: int
        :param controller_number: virtual machine's controller number.
        :type controller_number: int
        :param profile: A VMware Storage Policy profile
        :type profile: pbm.profile.Profile
        :returns: VMware task object.
        :rtype: Task
        """
        spec = vim.vm.ConfigSpec()
        profileSpec = vim.vm.DefinedProfileSpec()
        profileSpec.profileId = profile.profileId.uniqueId
        deviceSpec = vim.vm.device.VirtualDeviceSpec()
        deviceSpec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
        disk_obj = self.GetVirtualDiskObj(vm, unit_number, controller_number)
        deviceSpec.device = disk_obj
        deviceSpec.profile = [profileSpec]
        spec.deviceChange = [deviceSpec]
        return vm.ReconfigVM_Task(spec)

    def ensure_storage_policies(self, vm_obj):
        """
        Ensure VM storage profile policies.

        :param vm_obj: VMware VM object.
        :type vm_obj: VirtualMachine
        :exits: self.module.exit_json on success, else self.module.fail_json.
        """
        disks = self.module.params.get('disk')
        vm_home = self.module.params.get('vm_home')
        success_msg = 'Policies successfully set.'
        result = dict(changed=False, msg='', changed_policies=dict(disk=[], vm_home=''))
        self.get_spbm_connection()
        pm = self.spbm_content.profileManager
        if vm_home:
            policy = vm_home
            pmObjectType = pbm.ServerObjectRef.ObjectType('virtualMachine')
            pmRef = pbm.ServerObjectRef(key=vm_obj._moId, objectType=pmObjectType)
            pol_obj = self.SearchStorageProfileByName(pm, policy)
            if not pol_obj:
                result['msg'] = "Unable to find storage policy `%s' for vm_home" % policy
                self.module.fail_json(**result)
            if not self.CheckAssociatedStorageProfile(pm, pmRef, policy):
                if not self.module.check_mode:
                    task = self.SetVMHomeStorageProfile(vm_obj, pol_obj)
                    wait_for_task(task)
                result['changed'] = True
                result['changed_policies']['vm_home'] = policy
        if disks is None:
            disks = list()
        disks_objs = dict()
        for disk in disks:
            policy = str(disk['policy'])
            unit_number = int(disk['unit_number'])
            controller_number = int(disk['controller_number'])
            disk_obj = self.GetVirtualDiskObj(vm_obj, unit_number, controller_number)
            pol_obj = self.SearchStorageProfileByName(pm, policy)
            if not pol_obj:
                result['msg'] = "Unable to find storage policy `%s' for disk %s." % (policy, disk)
                self.module.fail_json(**result)
            if not disk_obj:
                errmsg = "Unable to find disk for controller_number '%s' unit_number '%s'. 7 is reserved for SCSI adapters."
                result['msg'] = errmsg % (controller_number, unit_number)
                self.module.fail_json(**result)
            disks_objs[unit_number] = dict(disk=disk_obj, policy=pol_obj)
        for disk in disks:
            policy = str(disk['policy'])
            unit_number = int(disk['unit_number'])
            controller_number = int(disk['controller_number'])
            disk_obj = disks_objs[unit_number]['disk']
            pol_obj = disks_objs[unit_number]['policy']
            pmObjectType = pbm.ServerObjectRef.ObjectType('virtualDiskId')
            pmRef = pbm.ServerObjectRef(key='%s:%s' % (vm_obj._moId, disk_obj.key), objectType=pmObjectType)
            if not self.CheckAssociatedStorageProfile(pm, pmRef, policy):
                if not self.module.check_mode:
                    task = self.SetVMDiskStorageProfile(vm_obj, unit_number, controller_number, pol_obj)
                    wait_for_task(task)
                result['changed'] = True
                result['changed_policies']['disk'].append(disk)
        if result['changed']:
            result['msg'] = success_msg
        self.module.exit_json(**result)