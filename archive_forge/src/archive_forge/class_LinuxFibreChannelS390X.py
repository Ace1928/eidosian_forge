from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
class LinuxFibreChannelS390X(LinuxFibreChannel):

    def get_fc_hbas_info(self):
        """Get Fibre Channel WWNs and device paths from the system, if any."""
        hbas = self.get_fc_hbas()
        hbas_info = []
        for hba in hbas:
            if hba['port_state'] == 'Online':
                wwpn = hba['port_name'].replace('0x', '')
                wwnn = hba['node_name'].replace('0x', '')
                device_path = hba['ClassDevicepath']
                device = hba['ClassDevice']
                hbas_info.append({'port_name': wwpn, 'node_name': wwnn, 'host_device': device, 'device_path': device_path})
        return hbas_info

    def configure_scsi_device(self, device_number, target_wwn, lun):
        """Write the LUN to the port's unit_add attribute.

        If auto-discovery of Fibre-Channel target ports is
        disabled on s390 platforms, ports need to be added to
        the configuration.
        If auto-discovery of LUNs is disabled on s390 platforms
        luns need to be added to the configuration through the
        unit_add interface
        """
        LOG.debug('Configure lun for s390: device_number=%(device_num)s target_wwn=%(target_wwn)s target_lun=%(target_lun)s', {'device_num': device_number, 'target_wwn': target_wwn, 'target_lun': lun})
        filepath = '/sys/bus/ccw/drivers/zfcp/%s/%s' % (device_number, target_wwn)
        if not os.path.exists(filepath):
            zfcp_device_command = '/sys/bus/ccw/drivers/zfcp/%s/port_rescan' % device_number
            LOG.debug('port_rescan call for s390: %s', zfcp_device_command)
            try:
                self.echo_scsi_command(zfcp_device_command, '1')
            except putils.ProcessExecutionError as exc:
                LOG.warning('port_rescan call for s390 failed exit %(code)s, stderr %(stderr)s', {'code': exc.exit_code, 'stderr': exc.stderr})
        zfcp_device_command = '/sys/bus/ccw/drivers/zfcp/%s/%s/unit_add' % (device_number, target_wwn)
        LOG.debug('unit_add call for s390 execute: %s', zfcp_device_command)
        try:
            self.echo_scsi_command(zfcp_device_command, lun)
        except putils.ProcessExecutionError as exc:
            LOG.warning('unit_add call for s390 failed exit %(code)s, stderr %(stderr)s', {'code': exc.exit_code, 'stderr': exc.stderr})

    def deconfigure_scsi_device(self, device_number, target_wwn, lun):
        """Write the LUN to the port's unit_remove attribute.

        If auto-discovery of LUNs is disabled on s390 platforms
        luns need to be removed from the configuration through the
        unit_remove interface
        """
        LOG.debug('Deconfigure lun for s390: device_number=%(device_num)s target_wwn=%(target_wwn)s target_lun=%(target_lun)s', {'device_num': device_number, 'target_wwn': target_wwn, 'target_lun': lun})
        zfcp_device_command = '/sys/bus/ccw/drivers/zfcp/%s/%s/unit_remove' % (device_number, target_wwn)
        LOG.debug('unit_remove call for s390 execute: %s', zfcp_device_command)
        try:
            self.echo_scsi_command(zfcp_device_command, lun)
        except putils.ProcessExecutionError as exc:
            LOG.warning('unit_remove call for s390 failed exit %(code)s, stderr %(stderr)s', {'code': exc.exit_code, 'stderr': exc.stderr})