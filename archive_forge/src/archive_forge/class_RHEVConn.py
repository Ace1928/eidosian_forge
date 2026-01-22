from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
class RHEVConn(object):
    """Connection to RHEV-M"""

    def __init__(self, module):
        self.module = module
        user = module.params.get('user')
        password = module.params.get('password')
        server = module.params.get('server')
        port = module.params.get('port')
        insecure_api = module.params.get('insecure_api')
        url = 'https://%s:%s' % (server, port)
        try:
            api = API(url=url, username=user, password=password, insecure=str(insecure_api))
            api.test()
            self.conn = api
        except Exception:
            raise Exception('Failed to connect to RHEV-M.')

    def __del__(self):
        self.conn.disconnect()

    def createVMimage(self, name, cluster, template):
        try:
            vmparams = params.VM(name=name, cluster=self.conn.clusters.get(name=cluster), template=self.conn.templates.get(name=template), disks=params.Disks(clone=True))
            self.conn.vms.add(vmparams)
            setMsg('VM is created')
            setChanged()
            return True
        except Exception as e:
            setMsg('Failed to create VM')
            setMsg(str(e))
            setFailed()
            return False

    def createVM(self, name, cluster, os, actiontype):
        try:
            vmparams = params.VM(name=name, cluster=self.conn.clusters.get(name=cluster), os=params.OperatingSystem(type_=os), template=self.conn.templates.get(name='Blank'), type_=actiontype)
            self.conn.vms.add(vmparams)
            setMsg('VM is created')
            setChanged()
            return True
        except Exception as e:
            setMsg('Failed to create VM')
            setMsg(str(e))
            setFailed()
            return False

    def createDisk(self, vmname, diskname, disksize, diskdomain, diskinterface, diskformat, diskallocationtype, diskboot):
        VM = self.get_VM(vmname)
        newdisk = params.Disk(name=diskname, size=1024 * 1024 * 1024 * int(disksize), wipe_after_delete=True, sparse=diskallocationtype, interface=diskinterface, format=diskformat, bootable=diskboot, storage_domains=params.StorageDomains(storage_domain=[self.get_domain(diskdomain)]))
        try:
            VM.disks.add(newdisk)
            VM.update()
            setMsg('Successfully added disk ' + diskname)
            setChanged()
        except Exception as e:
            setFailed()
            setMsg('Error attaching ' + diskname + 'disk, please recheck and remove any leftover configuration.')
            setMsg(str(e))
            return False
        try:
            currentdisk = VM.disks.get(name=diskname)
            attempt = 1
            while currentdisk.status.state != 'ok':
                currentdisk = VM.disks.get(name=diskname)
                if attempt == 100:
                    setMsg('Error, disk %s, state %s' % (diskname, str(currentdisk.status.state)))
                    raise Exception()
                else:
                    attempt += 1
                    time.sleep(2)
            setMsg('The disk  ' + diskname + ' is ready.')
        except Exception as e:
            setFailed()
            setMsg('Error getting the state of ' + diskname + '.')
            setMsg(str(e))
            return False
        return True

    def createNIC(self, vmname, nicname, vlan, interface):
        VM = self.get_VM(vmname)
        CLUSTER = self.get_cluster_byid(VM.cluster.id)
        DC = self.get_DC_byid(CLUSTER.data_center.id)
        newnic = params.NIC(name=nicname, network=DC.networks.get(name=vlan), interface=interface)
        try:
            VM.nics.add(newnic)
            VM.update()
            setMsg('Successfully added iface ' + nicname)
            setChanged()
        except Exception as e:
            setFailed()
            setMsg('Error attaching ' + nicname + ' iface, please recheck and remove any leftover configuration.')
            setMsg(str(e))
            return False
        try:
            currentnic = VM.nics.get(name=nicname)
            attempt = 1
            while currentnic.active is not True:
                currentnic = VM.nics.get(name=nicname)
                if attempt == 100:
                    setMsg('Error, iface %s, state %s' % (nicname, str(currentnic.active)))
                    raise Exception()
                else:
                    attempt += 1
                    time.sleep(2)
            setMsg('The iface  ' + nicname + ' is ready.')
        except Exception as e:
            setFailed()
            setMsg('Error getting the state of ' + nicname + '.')
            setMsg(str(e))
            return False
        return True

    def get_DC(self, dc_name):
        return self.conn.datacenters.get(name=dc_name)

    def get_DC_byid(self, dc_id):
        return self.conn.datacenters.get(id=dc_id)

    def get_VM(self, vm_name):
        return self.conn.vms.get(name=vm_name)

    def get_cluster_byid(self, cluster_id):
        return self.conn.clusters.get(id=cluster_id)

    def get_cluster(self, cluster_name):
        return self.conn.clusters.get(name=cluster_name)

    def get_domain_byid(self, dom_id):
        return self.conn.storagedomains.get(id=dom_id)

    def get_domain(self, domain_name):
        return self.conn.storagedomains.get(name=domain_name)

    def get_disk(self, disk):
        return self.conn.disks.get(disk)

    def get_network(self, dc_name, network_name):
        return self.get_DC(dc_name).networks.get(network_name)

    def get_network_byid(self, network_id):
        return self.conn.networks.get(id=network_id)

    def get_NIC(self, vm_name, nic_name):
        return self.get_VM(vm_name).nics.get(nic_name)

    def get_Host(self, host_name):
        return self.conn.hosts.get(name=host_name)

    def get_Host_byid(self, host_id):
        return self.conn.hosts.get(id=host_id)

    def set_Memory(self, name, memory):
        VM = self.get_VM(name)
        VM.memory = int(int(memory) * 1024 * 1024 * 1024)
        try:
            VM.update()
            setMsg('The Memory has been updated.')
            setChanged()
            return True
        except Exception as e:
            setMsg('Failed to update memory.')
            setMsg(str(e))
            setFailed()
            return False

    def set_Memory_Policy(self, name, memory_policy):
        VM = self.get_VM(name)
        VM.memory_policy.guaranteed = int(memory_policy) * 1024 * 1024 * 1024
        try:
            VM.update()
            setMsg('The memory policy has been updated.')
            setChanged()
            return True
        except Exception as e:
            setMsg('Failed to update memory policy.')
            setMsg(str(e))
            setFailed()
            return False

    def set_CPU(self, name, cpu):
        VM = self.get_VM(name)
        VM.cpu.topology.cores = int(cpu)
        try:
            VM.update()
            setMsg('The number of CPUs has been updated.')
            setChanged()
            return True
        except Exception as e:
            setMsg('Failed to update the number of CPUs.')
            setMsg(str(e))
            setFailed()
            return False

    def set_CPU_share(self, name, cpu_share):
        VM = self.get_VM(name)
        VM.cpu_shares = int(cpu_share)
        try:
            VM.update()
            setMsg('The CPU share has been updated.')
            setChanged()
            return True
        except Exception as e:
            setMsg('Failed to update the CPU share.')
            setMsg(str(e))
            setFailed()
            return False

    def set_Disk(self, diskname, disksize, diskinterface, diskboot):
        DISK = self.get_disk(diskname)
        setMsg('Checking disk ' + diskname)
        if DISK.get_bootable() != diskboot:
            try:
                DISK.set_bootable(diskboot)
                setMsg('Updated the boot option on the disk.')
                setChanged()
            except Exception as e:
                setMsg('Failed to set the boot option on the disk.')
                setMsg(str(e))
                setFailed()
                return False
        else:
            setMsg('The boot option of the disk is correct')
        if int(DISK.size) < 1024 * 1024 * 1024 * int(disksize):
            try:
                DISK.size = 1024 * 1024 * 1024 * int(disksize)
                setMsg('Updated the size of the disk.')
                setChanged()
            except Exception as e:
                setMsg('Failed to update the size of the disk.')
                setMsg(str(e))
                setFailed()
                return False
        elif int(DISK.size) > 1024 * 1024 * 1024 * int(disksize):
            setMsg('Shrinking disks is not supported')
            setFailed()
            return False
        else:
            setMsg('The size of the disk is correct')
        if str(DISK.interface) != str(diskinterface):
            try:
                DISK.interface = diskinterface
                setMsg('Updated the interface of the disk.')
                setChanged()
            except Exception as e:
                setMsg('Failed to update the interface of the disk.')
                setMsg(str(e))
                setFailed()
                return False
        else:
            setMsg('The interface of the disk is correct')
        return True

    def set_NIC(self, vmname, nicname, newname, vlan, interface):
        NIC = self.get_NIC(vmname, nicname)
        VM = self.get_VM(vmname)
        CLUSTER = self.get_cluster_byid(VM.cluster.id)
        DC = self.get_DC_byid(CLUSTER.data_center.id)
        NETWORK = self.get_network(str(DC.name), vlan)
        checkFail()
        if NIC.name != newname:
            NIC.name = newname
            setMsg('Updating iface name to ' + newname)
            setChanged()
        if str(NIC.network.id) != str(NETWORK.id):
            NIC.set_network(NETWORK)
            setMsg('Updating iface network to ' + vlan)
            setChanged()
        if NIC.interface != interface:
            NIC.interface = interface
            setMsg('Updating iface interface to ' + interface)
            setChanged()
        try:
            NIC.update()
            setMsg('iface has successfully been updated.')
        except Exception as e:
            setMsg('Failed to update the iface.')
            setMsg(str(e))
            setFailed()
            return False
        return True

    def set_DeleteProtection(self, vmname, del_prot):
        VM = self.get_VM(vmname)
        VM.delete_protected = del_prot
        try:
            VM.update()
            setChanged()
        except Exception as e:
            setMsg('Failed to update delete protection.')
            setMsg(str(e))
            setFailed()
            return False
        return True

    def set_BootOrder(self, vmname, boot_order):
        VM = self.get_VM(vmname)
        bootorder = []
        for device in boot_order:
            bootorder.append(params.Boot(dev=device))
        VM.os.boot = bootorder
        try:
            VM.update()
            setChanged()
        except Exception as e:
            setMsg('Failed to update the boot order.')
            setMsg(str(e))
            setFailed()
            return False
        return True

    def set_Host(self, host_name, cluster, ifaces):
        HOST = self.get_Host(host_name)
        CLUSTER = self.get_cluster(cluster)
        if HOST is None:
            setMsg('Host does not exist.')
            ifacelist = dict()
            networklist = []
            manageip = ''
            try:
                for iface in ifaces:
                    try:
                        setMsg('creating host interface ' + iface['name'])
                        if 'management' in iface:
                            manageip = iface['ip']
                        if 'boot_protocol' not in iface:
                            if 'ip' in iface:
                                iface['boot_protocol'] = 'static'
                            else:
                                iface['boot_protocol'] = 'none'
                        if 'ip' not in iface:
                            iface['ip'] = ''
                        if 'netmask' not in iface:
                            iface['netmask'] = ''
                        if 'gateway' not in iface:
                            iface['gateway'] = ''
                        if 'network' in iface:
                            if 'bond' in iface:
                                bond = []
                                for slave in iface['bond']:
                                    bond.append(ifacelist[slave])
                                try:
                                    tmpiface = params.Bonding(slaves=params.Slaves(host_nic=bond), options=params.Options(option=[params.Option(name='miimon', value='100'), params.Option(name='mode', value='4')]))
                                except Exception as e:
                                    setMsg('Failed to create the bond for  ' + iface['name'])
                                    setFailed()
                                    setMsg(str(e))
                                    return False
                                try:
                                    tmpnetwork = params.HostNIC(network=params.Network(name=iface['network']), name=iface['name'], boot_protocol=iface['boot_protocol'], ip=params.IP(address=iface['ip'], netmask=iface['netmask'], gateway=iface['gateway']), override_configuration=True, bonding=tmpiface)
                                    networklist.append(tmpnetwork)
                                    setMsg('Applying network ' + iface['name'])
                                except Exception as e:
                                    setMsg('Failed to set' + iface['name'] + ' as network interface')
                                    setFailed()
                                    setMsg(str(e))
                                    return False
                            else:
                                tmpnetwork = params.HostNIC(network=params.Network(name=iface['network']), name=iface['name'], boot_protocol=iface['boot_protocol'], ip=params.IP(address=iface['ip'], netmask=iface['netmask'], gateway=iface['gateway']))
                                networklist.append(tmpnetwork)
                                setMsg('Applying network ' + iface['name'])
                        else:
                            tmpiface = params.HostNIC(name=iface['name'], network=params.Network(), boot_protocol=iface['boot_protocol'], ip=params.IP(address=iface['ip'], netmask=iface['netmask'], gateway=iface['gateway']))
                        ifacelist[iface['name']] = tmpiface
                    except Exception as e:
                        setMsg('Failed to set ' + iface['name'])
                        setFailed()
                        setMsg(str(e))
                        return False
            except Exception as e:
                setMsg('Failed to set networks')
                setMsg(str(e))
                setFailed()
                return False
            if manageip == '':
                setMsg('No management network is defined')
                setFailed()
                return False
            try:
                HOST = params.Host(name=host_name, address=manageip, cluster=CLUSTER, ssh=params.SSH(authentication_method='publickey'))
                if self.conn.hosts.add(HOST):
                    setChanged()
                    HOST = self.get_Host(host_name)
                    state = HOST.status.state
                    while state != 'non_operational' and state != 'up':
                        HOST = self.get_Host(host_name)
                        state = HOST.status.state
                        time.sleep(1)
                        if state == 'non_responsive':
                            setMsg('Failed to add host to RHEVM')
                            setFailed()
                            return False
                    setMsg('status host: up')
                    time.sleep(5)
                    HOST = self.get_Host(host_name)
                    state = HOST.status.state
                    setMsg('State before setting to maintenance: ' + str(state))
                    HOST.deactivate()
                    while state != 'maintenance':
                        HOST = self.get_Host(host_name)
                        state = HOST.status.state
                        time.sleep(1)
                    setMsg('status host: maintenance')
                    try:
                        HOST.nics.setupnetworks(params.Action(force=True, check_connectivity=False, host_nics=params.HostNics(host_nic=networklist)))
                        setMsg('nics are set')
                    except Exception as e:
                        setMsg('Failed to apply networkconfig')
                        setFailed()
                        setMsg(str(e))
                        return False
                    try:
                        HOST.commitnetconfig()
                        setMsg('Network config is saved')
                    except Exception as e:
                        setMsg('Failed to save networkconfig')
                        setFailed()
                        setMsg(str(e))
                        return False
            except Exception as e:
                if 'The Host name is already in use' in str(e):
                    setMsg('Host already exists')
                else:
                    setMsg('Failed to add host')
                    setFailed()
                    setMsg(str(e))
                return False
            HOST.activate()
            while state != 'up':
                HOST = self.get_Host(host_name)
                state = HOST.status.state
                time.sleep(1)
                if state == 'non_responsive':
                    setMsg('Failed to apply networkconfig.')
                    setFailed()
                    return False
            setMsg('status host: up')
        else:
            setMsg('Host exists.')
        return True

    def del_NIC(self, vmname, nicname):
        return self.get_NIC(vmname, nicname).delete()

    def remove_VM(self, vmname):
        VM = self.get_VM(vmname)
        try:
            VM.delete()
        except Exception as e:
            setMsg('Failed to remove VM.')
            setMsg(str(e))
            setFailed()
            return False
        return True

    def start_VM(self, vmname, timeout):
        VM = self.get_VM(vmname)
        try:
            VM.start()
        except Exception as e:
            setMsg('Failed to start VM.')
            setMsg(str(e))
            setFailed()
            return False
        return self.wait_VM(vmname, 'up', timeout)

    def wait_VM(self, vmname, state, timeout):
        VM = self.get_VM(vmname)
        while VM.status.state != state:
            VM = self.get_VM(vmname)
            time.sleep(10)
            if timeout is not False:
                timeout -= 10
                if timeout <= 0:
                    setMsg('Timeout expired')
                    setFailed()
                    return False
        return True

    def stop_VM(self, vmname, timeout):
        VM = self.get_VM(vmname)
        try:
            VM.stop()
        except Exception as e:
            setMsg('Failed to stop VM.')
            setMsg(str(e))
            setFailed()
            return False
        return self.wait_VM(vmname, 'down', timeout)

    def set_CD(self, vmname, cd_drive):
        VM = self.get_VM(vmname)
        try:
            if str(VM.status.state) == 'down':
                cdrom = params.CdRom(file=cd_drive)
                VM.cdroms.add(cdrom)
                setMsg('Attached the image.')
                setChanged()
            else:
                cdrom = VM.cdroms.get(id='00000000-0000-0000-0000-000000000000')
                cdrom.set_file(cd_drive)
                cdrom.update(current=True)
                setMsg('Attached the image.')
                setChanged()
        except Exception as e:
            setMsg('Failed to attach image.')
            setMsg(str(e))
            setFailed()
            return False
        return True

    def set_VM_Host(self, vmname, vmhost):
        VM = self.get_VM(vmname)
        HOST = self.get_Host(vmhost)
        try:
            VM.placement_policy.host = HOST
            VM.update()
            setMsg('Set startup host to ' + vmhost)
            setChanged()
        except Exception as e:
            setMsg('Failed to set startup host.')
            setMsg(str(e))
            setFailed()
            return False
        return True

    def migrate_VM(self, vmname, vmhost):
        VM = self.get_VM(vmname)
        HOST = self.get_Host_byid(VM.host.id)
        if str(HOST.name) != vmhost:
            try:
                VM.migrate(action=params.Action(host=params.Host(name=vmhost)))
                setChanged()
                setMsg('VM migrated to ' + vmhost)
            except Exception as e:
                setMsg('Failed to set startup host.')
                setMsg(str(e))
                setFailed()
                return False
        return True

    def remove_CD(self, vmname):
        VM = self.get_VM(vmname)
        try:
            VM.cdroms.get(id='00000000-0000-0000-0000-000000000000').delete()
            setMsg('Removed the image.')
            setChanged()
        except Exception as e:
            setMsg('Failed to remove the image.')
            setMsg(str(e))
            setFailed()
            return False
        return True