from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
class SystemInfoParameters(BaseParameters):
    api_map = {'isSystemSetup': 'is_system_setup', 'isAdminPasswordChanged': 'is_admin_password_changed', 'isRootPasswordChanged': 'is_root_password_changed'}
    returnables = ['base_mac_address', 'chassis_serial', 'hardware_information', 'host_board_part_revision', 'host_board_serial', 'is_admin_password_changed', 'is_root_password_changed', 'is_system_setup', 'marketing_name', 'package_edition', 'package_version', 'platform', 'product_build', 'product_build_date', 'product_built', 'product_changelist', 'product_code', 'product_information', 'product_jobid', 'product_version', 'switch_board_part_revision', 'switch_board_serial', 'time', 'uptime']

    @property
    def is_admin_password_changed(self):
        return flatten_boolean(self._values['is_admin_password_changed'])

    @property
    def is_root_password_changed(self):
        return flatten_boolean(self._values['is_root_password_changed'])

    @property
    def is_system_setup(self):
        if self._values['is_system_setup'] is None:
            return 'no'
        return flatten_boolean(self._values['is_system_setup'])

    @property
    def chassis_serial(self):
        if self._values['system-info'] is None:
            return None
        if 'bigipChassisSerialNum' not in self._values['system-info'][0]:
            return None
        return self._values['system-info'][0]['bigipChassisSerialNum']

    @property
    def switch_board_serial(self):
        if self._values['system-info'] is None:
            return None
        if 'switchBoardSerialNum' not in self._values['system-info'][0]:
            return None
        if self._values['system-info'][0]['switchBoardSerialNum'].strip() == '':
            return None
        return self._values['system-info'][0]['switchBoardSerialNum']

    @property
    def switch_board_part_revision(self):
        if self._values['system-info'] is None:
            return None
        if 'switchBoardPartRevNum' not in self._values['system-info'][0]:
            return None
        if self._values['system-info'][0]['switchBoardPartRevNum'].strip() == '':
            return None
        return self._values['system-info'][0]['switchBoardPartRevNum']

    @property
    def platform(self):
        if self._values['system-info'] is None:
            return None
        return self._values['system-info'][0]['platform']

    @property
    def host_board_serial(self):
        if self._values['system-info'] is None:
            return None
        if 'hostBoardSerialNum' not in self._values['system-info'][0]:
            return None
        if self._values['system-info'][0]['hostBoardSerialNum'].strip() == '':
            return None
        return self._values['system-info'][0]['hostBoardSerialNum']

    @property
    def host_board_part_revision(self):
        if self._values['system-info'] is None:
            return None
        if 'hostBoardPartRevNum' not in self._values['system-info'][0]:
            return None
        if self._values['system-info'][0]['hostBoardPartRevNum'].strip() == '':
            return None
        return self._values['system-info'][0]['hostBoardPartRevNum']

    @property
    def package_edition(self):
        return self._values['Edition']

    @property
    def package_version(self):
        return 'Build {0} - {1}'.format(self._values['Build'], self._values['Date'])

    @property
    def product_build(self):
        return self._values['Build']

    @property
    def product_build_date(self):
        return self._values['Date']

    @property
    def product_built(self):
        if 'version_info' not in self._values:
            return None
        if 'Built' in self._values['version_info']:
            return int(self._values['version_info']['Built'])

    @property
    def product_changelist(self):
        if 'version_info' not in self._values:
            return None
        if 'Changelist' in self._values['version_info']:
            return int(self._values['version_info']['Changelist'])

    @property
    def product_jobid(self):
        if 'version_info' not in self._values:
            return None
        if 'JobID' in self._values['version_info']:
            return int(self._values['version_info']['JobID'])

    @property
    def product_code(self):
        return self._values['Product']

    @property
    def product_version(self):
        return self._values['Version']

    @property
    def hardware_information(self):
        if self._values['hardware-version'] is None:
            return None
        self._transform_name_attribute(self._values['hardware-version'])
        result = [v for k, v in iteritems(self._values['hardware-version'])]
        return result

    def _transform_name_attribute(self, entry):
        if isinstance(entry, dict):
            tmp = copy.deepcopy(entry)
            for k, v in iteritems(tmp):
                if k == 'tmName':
                    entry['name'] = entry.pop('tmName')
                self._transform_name_attribute(v)
        elif isinstance(entry, list):
            for k in entry:
                self._transform_name_attribute(k)
        else:
            return

    @property
    def time(self):
        if self._values['fullDate'] is None:
            return None
        date = datetime.datetime.strptime(self._values['fullDate'], '%Y-%m-%dT%H:%M:%SZ')
        result = dict(day=date.day, hour=date.hour, minute=date.minute, month=date.month, second=date.second, year=date.year)
        return result

    @property
    def marketing_name(self):
        if self._values['platform'] is None:
            return None
        return self._values['platform'][0]['marketingName']

    @property
    def base_mac_address(self):
        if self._values['platform'] is None:
            return None
        return self._values['platform'][0]['baseMac']