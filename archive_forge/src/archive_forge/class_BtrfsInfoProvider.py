from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
class BtrfsInfoProvider(object):
    """
    Utility providing details of the currently available btrfs filesystems
    """

    def __init__(self, module):
        self.__module = module
        self.__btrfs_api = BtrfsCommands(module)
        self.__findmnt_path = self.__module.get_bin_path('findmnt', required=True)

    def get_filesystems(self):
        filesystems = self.__btrfs_api.filesystem_show()
        mountpoints = self.__find_mountpoints()
        for filesystem in filesystems:
            device_mountpoints = self.__filter_mountpoints_for_devices(mountpoints, filesystem['devices'])
            filesystem['mountpoints'] = device_mountpoints
            if len(device_mountpoints) > 0:
                mountpoint = device_mountpoints[0]['mountpoint']
                filesystem['subvolumes'] = self.get_subvolumes(mountpoint)
                filesystem['default_subvolid'] = self.get_default_subvolume_id(mountpoint)
        return filesystems

    def get_mountpoints(self, filesystem_devices):
        mountpoints = self.__find_mountpoints()
        return self.__filter_mountpoints_for_devices(mountpoints, filesystem_devices)

    def get_subvolumes(self, filesystem_path):
        return self.__btrfs_api.subvolumes_list(filesystem_path)

    def get_default_subvolume_id(self, filesystem_path):
        return self.__btrfs_api.subvolume_get_default(filesystem_path)

    def __filter_mountpoints_for_devices(self, mountpoints, devices):
        return [m for m in mountpoints if m['device'] in devices]

    def __find_mountpoints(self):
        command = '%s -t btrfs -nvP' % self.__findmnt_path
        result = self.__module.run_command(command)
        mountpoints = []
        if result[0] == 0:
            lines = result[1].splitlines()
            for line in lines:
                mountpoint = self.__parse_mountpoint_pairs(line)
                mountpoints.append(mountpoint)
        return mountpoints

    def __parse_mountpoint_pairs(self, line):
        pattern = re.compile('^TARGET="(?P<target>.*)"\\s+SOURCE="(?P<source>.*)"\\s+FSTYPE="(?P<fstype>.*)"\\s+OPTIONS="(?P<options>.*)"\\s*$')
        match = pattern.search(line)
        if match is not None:
            groups = match.groupdict()
            return {'mountpoint': groups['target'], 'device': groups['source'], 'subvolid': self.__extract_mount_subvolid(groups['options'])}
        else:
            raise BtrfsModuleException("Failed to parse findmnt result for line: '%s'" % line)

    def __extract_mount_subvolid(self, mount_options):
        for option in mount_options.split(','):
            if option.startswith('subvolid='):
                return int(option[len('subvolid='):])
        raise BtrfsModuleException("Failed to find subvolid for mountpoint in options '%s'" % mount_options)