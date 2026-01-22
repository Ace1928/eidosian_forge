from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
class RHEV(object):

    def __init__(self, module):
        self.module = module

    def __get_conn(self):
        self.conn = RHEVConn(self.module)
        return self.conn

    def test(self):
        self.__get_conn()
        return 'OK'

    def getVM(self, name):
        self.__get_conn()
        VM = self.conn.get_VM(name)
        if VM:
            vminfo = dict()
            vminfo['uuid'] = VM.id
            vminfo['name'] = VM.name
            vminfo['status'] = VM.status.state
            vminfo['cpu_cores'] = VM.cpu.topology.cores
            vminfo['cpu_sockets'] = VM.cpu.topology.sockets
            vminfo['cpu_shares'] = VM.cpu_shares
            vminfo['memory'] = int(VM.memory) // 1024 // 1024 // 1024
            vminfo['mem_pol'] = int(VM.memory_policy.guaranteed) // 1024 // 1024 // 1024
            vminfo['os'] = VM.get_os().type_
            vminfo['del_prot'] = VM.delete_protected
            try:
                vminfo['host'] = str(self.conn.get_Host_byid(str(VM.host.id)).name)
            except Exception:
                vminfo['host'] = None
            vminfo['boot_order'] = []
            for boot_dev in VM.os.get_boot():
                vminfo['boot_order'].append(str(boot_dev.dev))
            vminfo['disks'] = []
            for DISK in VM.disks.list():
                disk = dict()
                disk['name'] = DISK.name
                disk['size'] = int(DISK.size) // 1024 // 1024 // 1024
                disk['domain'] = str(self.conn.get_domain_byid(DISK.get_storage_domains().get_storage_domain()[0].id).name)
                disk['interface'] = DISK.interface
                vminfo['disks'].append(disk)
            vminfo['ifaces'] = []
            for NIC in VM.nics.list():
                iface = dict()
                iface['name'] = str(NIC.name)
                iface['vlan'] = str(self.conn.get_network_byid(NIC.get_network().id).name)
                iface['interface'] = NIC.interface
                iface['mac'] = NIC.mac.address
                vminfo['ifaces'].append(iface)
                vminfo[str(NIC.name)] = NIC.mac.address
            CLUSTER = self.conn.get_cluster_byid(VM.cluster.id)
            if CLUSTER:
                vminfo['cluster'] = CLUSTER.name
        else:
            vminfo = False
        return vminfo

    def createVMimage(self, name, cluster, template, disks):
        self.__get_conn()
        return self.conn.createVMimage(name, cluster, template, disks)

    def createVM(self, name, cluster, os, actiontype):
        self.__get_conn()
        return self.conn.createVM(name, cluster, os, actiontype)

    def setMemory(self, name, memory):
        self.__get_conn()
        return self.conn.set_Memory(name, memory)

    def setMemoryPolicy(self, name, memory_policy):
        self.__get_conn()
        return self.conn.set_Memory_Policy(name, memory_policy)

    def setCPU(self, name, cpu):
        self.__get_conn()
        return self.conn.set_CPU(name, cpu)

    def setCPUShare(self, name, cpu_share):
        self.__get_conn()
        return self.conn.set_CPU_share(name, cpu_share)

    def setDisks(self, name, disks):
        self.__get_conn()
        counter = 0
        bootselect = False
        for disk in disks:
            if 'bootable' in disk:
                if disk['bootable'] is True:
                    bootselect = True
        for disk in disks:
            diskname = name + '_Disk' + str(counter) + '_' + disk.get('name', '').replace('/', '_')
            disksize = disk.get('size', 1)
            diskdomain = disk.get('domain', None)
            if diskdomain is None:
                setMsg('`domain` is a required disk key.')
                setFailed()
                return False
            diskinterface = disk.get('interface', 'virtio')
            diskformat = disk.get('format', 'raw')
            diskallocationtype = disk.get('thin', False)
            diskboot = disk.get('bootable', False)
            if bootselect is False and counter == 0:
                diskboot = True
            DISK = self.conn.get_disk(diskname)
            if DISK is None:
                self.conn.createDisk(name, diskname, disksize, diskdomain, diskinterface, diskformat, diskallocationtype, diskboot)
            else:
                self.conn.set_Disk(diskname, disksize, diskinterface, diskboot)
            checkFail()
            counter += 1
        return True

    def setNetworks(self, vmname, ifaces):
        self.__get_conn()
        VM = self.conn.get_VM(vmname)
        counter = 0
        length = len(ifaces)
        for NIC in VM.nics.list():
            if counter < length:
                iface = ifaces[counter]
                name = iface.get('name', None)
                if name is None:
                    setMsg('`name` is a required iface key.')
                    setFailed()
                elif str(name) != str(NIC.name):
                    setMsg('ifaces are in the wrong order, rebuilding everything.')
                    for NIC in VM.nics.list():
                        self.conn.del_NIC(vmname, NIC.name)
                    self.setNetworks(vmname, ifaces)
                    checkFail()
                    return True
                vlan = iface.get('vlan', None)
                if vlan is None:
                    setMsg('`vlan` is a required iface key.')
                    setFailed()
                checkFail()
                interface = iface.get('interface', 'virtio')
                self.conn.set_NIC(vmname, str(NIC.name), name, vlan, interface)
            else:
                self.conn.del_NIC(vmname, NIC.name)
            counter += 1
            checkFail()
        while counter < length:
            iface = ifaces[counter]
            name = iface.get('name', None)
            if name is None:
                setMsg('`name` is a required iface key.')
                setFailed()
            vlan = iface.get('vlan', None)
            if vlan is None:
                setMsg('`vlan` is a required iface key.')
                setFailed()
            if failed is True:
                return False
            interface = iface.get('interface', 'virtio')
            self.conn.createNIC(vmname, name, vlan, interface)
            counter += 1
            checkFail()
        return True

    def setDeleteProtection(self, vmname, del_prot):
        self.__get_conn()
        VM = self.conn.get_VM(vmname)
        if bool(VM.delete_protected) != bool(del_prot):
            self.conn.set_DeleteProtection(vmname, del_prot)
            checkFail()
            setMsg('`delete protection` has been updated.')
        else:
            setMsg('`delete protection` already has the right value.')
        return True

    def setBootOrder(self, vmname, boot_order):
        self.__get_conn()
        VM = self.conn.get_VM(vmname)
        bootorder = []
        for boot_dev in VM.os.get_boot():
            bootorder.append(str(boot_dev.dev))
        if boot_order != bootorder:
            self.conn.set_BootOrder(vmname, boot_order)
            setMsg('The boot order has been set')
        else:
            setMsg('The boot order has already been set')
        return True

    def removeVM(self, vmname):
        self.__get_conn()
        self.setPower(vmname, 'down', 300)
        return self.conn.remove_VM(vmname)

    def setPower(self, vmname, state, timeout):
        self.__get_conn()
        VM = self.conn.get_VM(vmname)
        if VM is None:
            setMsg('VM does not exist.')
            setFailed()
            return False
        if state == VM.status.state:
            setMsg('VM state was already ' + state)
        else:
            if state == 'up':
                setMsg('VM is going to start')
                self.conn.start_VM(vmname, timeout)
                setChanged()
            elif state == 'down':
                setMsg('VM is going to stop')
                self.conn.stop_VM(vmname, timeout)
                setChanged()
            elif state == 'restarted':
                self.setPower(vmname, 'down', timeout)
                checkFail()
                self.setPower(vmname, 'up', timeout)
            checkFail()
            setMsg('the vm state is set to ' + state)
        return True

    def setCD(self, vmname, cd_drive):
        self.__get_conn()
        if cd_drive:
            return self.conn.set_CD(vmname, cd_drive)
        else:
            return self.conn.remove_CD(vmname)

    def setVMHost(self, vmname, vmhost):
        self.__get_conn()
        return self.conn.set_VM_Host(vmname, vmhost)

    def setHost(self, hostname, cluster, ifaces):
        self.__get_conn()
        return self.conn.set_Host(hostname, cluster, ifaces)