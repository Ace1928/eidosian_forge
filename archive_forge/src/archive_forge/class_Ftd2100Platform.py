from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
class Ftd2100Platform(AbstractFtdPlatform):
    PLATFORM_MODELS = [FtdModel.FTD_2110, FtdModel.FTD_2120, FtdModel.FTD_2130, FtdModel.FTD_2140]

    def __init__(self, params):
        self._ftd = Kp(hostname=params['device_hostname'], login_username=params['device_username'], login_password=params['device_password'], sudo_password=params.get('device_sudo_password') or params['device_password'])

    def install_ftd_image(self, params):
        line = self._ftd.ssh_console(ip=params['console_ip'], port=params['console_port'], username=params['console_username'], password=params['console_password'])
        try:
            rommon_server, rommon_path = self.parse_rommon_file_location(params['rommon_file_location'])
            line.baseline_fp2k_ftd(tftp_server=rommon_server, rommon_file=rommon_path, uut_hostname=params['device_hostname'], uut_username=params['device_username'], uut_password=params.get('device_new_password') or params['device_password'], uut_ip=params['device_ip'], uut_netmask=params['device_netmask'], uut_gateway=params['device_gateway'], dns_servers=params['dns_server'], search_domains=params['search_domains'], fxos_url=params['image_file_location'], ftd_version=params['image_version'])
        finally:
            line.disconnect()