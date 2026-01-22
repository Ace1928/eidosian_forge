from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
class DistributionFiles:
    """has-a various distro file parsers (os-release, etc) and logic for finding the right one."""
    OSDIST_LIST = ({'path': '/etc/altlinux-release', 'name': 'Altlinux'}, {'path': '/etc/oracle-release', 'name': 'OracleLinux'}, {'path': '/etc/slackware-version', 'name': 'Slackware'}, {'path': '/etc/centos-release', 'name': 'CentOS'}, {'path': '/etc/redhat-release', 'name': 'RedHat'}, {'path': '/etc/vmware-release', 'name': 'VMwareESX', 'allowempty': True}, {'path': '/etc/openwrt_release', 'name': 'OpenWrt'}, {'path': '/etc/os-release', 'name': 'Amazon'}, {'path': '/etc/system-release', 'name': 'Amazon'}, {'path': '/etc/alpine-release', 'name': 'Alpine'}, {'path': '/etc/arch-release', 'name': 'Archlinux', 'allowempty': True}, {'path': '/etc/os-release', 'name': 'Archlinux'}, {'path': '/etc/os-release', 'name': 'SUSE'}, {'path': '/etc/SuSE-release', 'name': 'SUSE'}, {'path': '/etc/gentoo-release', 'name': 'Gentoo'}, {'path': '/etc/os-release', 'name': 'Debian'}, {'path': '/etc/lsb-release', 'name': 'Debian'}, {'path': '/etc/lsb-release', 'name': 'Mandriva'}, {'path': '/etc/sourcemage-release', 'name': 'SMGL'}, {'path': '/usr/lib/os-release', 'name': 'ClearLinux'}, {'path': '/etc/coreos/update.conf', 'name': 'Coreos'}, {'path': '/etc/os-release', 'name': 'Flatcar'}, {'path': '/etc/os-release', 'name': 'NA'})
    SEARCH_STRING = {'OracleLinux': 'Oracle Linux', 'RedHat': 'Red Hat', 'Altlinux': 'ALT', 'SMGL': 'Source Mage GNU/Linux'}
    OS_RELEASE_ALIAS = {'Archlinux': 'Arch Linux'}
    STRIP_QUOTES = '\\\'\\"\\\\'

    def __init__(self, module):
        self.module = module

    def _get_file_content(self, path):
        return get_file_content(path)

    def _get_dist_file_content(self, path, allow_empty=False):
        if not _file_exists(path, allow_empty=allow_empty):
            return (False, None)
        data = self._get_file_content(path)
        return (True, data)

    def _parse_dist_file(self, name, dist_file_content, path, collected_facts):
        dist_file_dict = {}
        dist_file_content = dist_file_content.strip(DistributionFiles.STRIP_QUOTES)
        if name in self.SEARCH_STRING:
            if self.SEARCH_STRING[name] in dist_file_content:
                dist_file_dict['distribution'] = name
                dist_file_dict['distribution_file_search_string'] = self.SEARCH_STRING[name]
            else:
                dist_file_dict['distribution'] = dist_file_content.split()[0]
            return (True, dist_file_dict)
        if name in self.OS_RELEASE_ALIAS:
            if self.OS_RELEASE_ALIAS[name] in dist_file_content:
                dist_file_dict['distribution'] = name
                return (True, dist_file_dict)
            return (False, dist_file_dict)
        try:
            distfunc_name = 'parse_distribution_file_' + name
            distfunc = getattr(self, distfunc_name)
            parsed, dist_file_dict = distfunc(name, dist_file_content, path, collected_facts)
            return (parsed, dist_file_dict)
        except AttributeError as exc:
            self.module.debug('exc: %s' % exc)
            return (False, dist_file_dict)
        return (True, dist_file_dict)

    def _guess_distribution(self):
        dist = (get_distribution(), get_distribution_version(), get_distribution_codename())
        distribution_guess = {'distribution': dist[0] or 'NA', 'distribution_version': dist[1] or 'NA', 'distribution_release': 'NA' if dist[2] is None else dist[2]}
        distribution_guess['distribution_major_version'] = distribution_guess['distribution_version'].split('.')[0] or 'NA'
        return distribution_guess

    def process_dist_files(self):
        dist_file_facts = {}
        dist_guess = self._guess_distribution()
        dist_file_facts.update(dist_guess)
        for ddict in self.OSDIST_LIST:
            name = ddict['name']
            path = ddict['path']
            allow_empty = ddict.get('allowempty', False)
            has_dist_file, dist_file_content = self._get_dist_file_content(path, allow_empty=allow_empty)
            if has_dist_file and allow_empty:
                dist_file_facts['distribution'] = name
                dist_file_facts['distribution_file_path'] = path
                dist_file_facts['distribution_file_variety'] = name
                break
            if not has_dist_file:
                continue
            parsed_dist_file, parsed_dist_file_facts = self._parse_dist_file(name, dist_file_content, path, dist_file_facts)
            if parsed_dist_file:
                dist_file_facts['distribution'] = name
                dist_file_facts['distribution_file_path'] = path
                dist_file_facts['distribution_file_variety'] = name
                dist_file_facts['distribution_file_parsed'] = parsed_dist_file
                dist_file_facts.update(parsed_dist_file_facts)
                break
        return dist_file_facts

    def parse_distribution_file_Slackware(self, name, data, path, collected_facts):
        slackware_facts = {}
        if 'Slackware' not in data:
            return (False, slackware_facts)
        slackware_facts['distribution'] = name
        version = re.findall('\\w+[.]\\w+\\+?', data)
        if version:
            slackware_facts['distribution_version'] = version[0]
        return (True, slackware_facts)

    def parse_distribution_file_Amazon(self, name, data, path, collected_facts):
        amazon_facts = {}
        if 'Amazon' not in data:
            return (False, amazon_facts)
        amazon_facts['distribution'] = 'Amazon'
        if path == '/etc/os-release':
            version = re.search('VERSION_ID=\\"(.*)\\"', data)
            if version:
                distribution_version = version.group(1)
                amazon_facts['distribution_version'] = distribution_version
                version_data = distribution_version.split('.')
                if len(version_data) > 1:
                    major, minor = version_data
                else:
                    major, minor = (version_data[0], 'NA')
                amazon_facts['distribution_major_version'] = major
                amazon_facts['distribution_minor_version'] = minor
        else:
            version = [n for n in data.split() if n.isdigit()]
            version = version[0] if version else 'NA'
            amazon_facts['distribution_version'] = version
        return (True, amazon_facts)

    def parse_distribution_file_OpenWrt(self, name, data, path, collected_facts):
        openwrt_facts = {}
        if 'OpenWrt' not in data:
            return (False, openwrt_facts)
        openwrt_facts['distribution'] = name
        version = re.search('DISTRIB_RELEASE="(.*)"', data)
        if version:
            openwrt_facts['distribution_version'] = version.groups()[0]
        release = re.search('DISTRIB_CODENAME="(.*)"', data)
        if release:
            openwrt_facts['distribution_release'] = release.groups()[0]
        return (True, openwrt_facts)

    def parse_distribution_file_Alpine(self, name, data, path, collected_facts):
        alpine_facts = {}
        alpine_facts['distribution'] = 'Alpine'
        alpine_facts['distribution_version'] = data
        return (True, alpine_facts)

    def parse_distribution_file_SUSE(self, name, data, path, collected_facts):
        suse_facts = {}
        if 'suse' not in data.lower():
            return (False, suse_facts)
        if path == '/etc/os-release':
            for line in data.splitlines():
                distribution = re.search('^NAME=(.*)', line)
                if distribution:
                    suse_facts['distribution'] = distribution.group(1).strip('"')
                distribution_version = re.search('^VERSION_ID="?([0-9]+\\.?[0-9]*)"?', line)
                if distribution_version:
                    suse_facts['distribution_version'] = distribution_version.group(1)
                    suse_facts['distribution_major_version'] = distribution_version.group(1).split('.')[0]
                if 'open' in data.lower():
                    release = re.search('^VERSION_ID="?[0-9]+\\.?([0-9]*)"?', line)
                    if release:
                        suse_facts['distribution_release'] = release.groups()[0]
                elif 'enterprise' in data.lower() and 'VERSION_ID' in line:
                    release = re.search('^VERSION_ID="?[0-9]+\\.?([0-9]*)"?', line)
                    if release.group(1):
                        release = release.group(1)
                    else:
                        release = '0'
                    suse_facts['distribution_release'] = release
        elif path == '/etc/SuSE-release':
            if 'open' in data.lower():
                data = data.splitlines()
                distdata = get_file_content(path).splitlines()[0]
                suse_facts['distribution'] = distdata.split()[0]
                for line in data:
                    release = re.search('CODENAME *= *([^\n]+)', line)
                    if release:
                        suse_facts['distribution_release'] = release.groups()[0].strip()
            elif 'enterprise' in data.lower():
                lines = data.splitlines()
                distribution = lines[0].split()[0]
                if 'Server' in data:
                    suse_facts['distribution'] = 'SLES'
                elif 'Desktop' in data:
                    suse_facts['distribution'] = 'SLED'
                for line in lines:
                    release = re.search('PATCHLEVEL = ([0-9]+)', line)
                    if release:
                        suse_facts['distribution_release'] = release.group(1)
                        suse_facts['distribution_version'] = collected_facts['distribution_version'] + '.' + release.group(1)
        if os.path.islink('/etc/products.d/baseproduct') and os.path.realpath('/etc/products.d/baseproduct').endswith('SLES_SAP.prod'):
            suse_facts['distribution'] = 'SLES_SAP'
        return (True, suse_facts)

    def parse_distribution_file_Debian(self, name, data, path, collected_facts):
        debian_facts = {}
        if 'Debian' in data or 'Raspbian' in data:
            debian_facts['distribution'] = 'Debian'
            release = re.search('PRETTY_NAME=[^(]+ \\(?([^)]+?)\\)', data)
            if release:
                debian_facts['distribution_release'] = release.groups()[0]
            if collected_facts['distribution_release'] == 'NA' and 'Debian' in data:
                dpkg_cmd = self.module.get_bin_path('dpkg')
                if dpkg_cmd:
                    cmd = "%s --status tzdata|grep Provides|cut -f2 -d'-'" % dpkg_cmd
                    rc, out, err = self.module.run_command(cmd)
                    if rc == 0:
                        debian_facts['distribution_release'] = out.strip()
            debian_version_path = '/etc/debian_version'
            distdata = get_file_lines(debian_version_path)
            for line in distdata:
                m = re.search('(\\d+)\\.(\\d+)', line.strip())
                if m:
                    debian_facts['distribution_minor_version'] = m.groups()[1]
        elif 'Ubuntu' in data:
            debian_facts['distribution'] = 'Ubuntu'
        elif 'SteamOS' in data:
            debian_facts['distribution'] = 'SteamOS'
        elif path in ('/etc/lsb-release', '/etc/os-release') and ('Kali' in data or 'Parrot' in data):
            if 'Kali' in data:
                debian_facts['distribution'] = 'Kali'
            elif 'Parrot' in data:
                debian_facts['distribution'] = 'Parrot'
            release = re.search('DISTRIB_RELEASE=(.*)', data)
            if release:
                debian_facts['distribution_release'] = release.groups()[0]
        elif 'Devuan' in data:
            debian_facts['distribution'] = 'Devuan'
            release = re.search('PRETTY_NAME=\\"?[^(\\"]+ \\(?([^) \\"]+)\\)?', data)
            if release:
                debian_facts['distribution_release'] = release.groups()[0]
            version = re.search('VERSION_ID=\\"(.*)\\"', data)
            if version:
                debian_facts['distribution_version'] = version.group(1)
                debian_facts['distribution_major_version'] = version.group(1)
        elif 'Cumulus' in data:
            debian_facts['distribution'] = 'Cumulus Linux'
            version = re.search('VERSION_ID=(.*)', data)
            if version:
                major, _minor, _dummy_ver = version.group(1).split('.')
                debian_facts['distribution_version'] = version.group(1)
                debian_facts['distribution_major_version'] = major
            release = re.search('VERSION="(.*)"', data)
            if release:
                debian_facts['distribution_release'] = release.groups()[0]
        elif 'Mint' in data:
            debian_facts['distribution'] = 'Linux Mint'
            version = re.search('VERSION_ID=\\"(.*)\\"', data)
            if version:
                debian_facts['distribution_version'] = version.group(1)
                debian_facts['distribution_major_version'] = version.group(1).split('.')[0]
        elif 'UOS' in data or 'Uos' in data or 'uos' in data:
            debian_facts['distribution'] = 'Uos'
            release = re.search('VERSION_CODENAME=\\"?([^\\"]+)\\"?', data)
            if release:
                debian_facts['distribution_release'] = release.groups()[0]
            version = re.search('VERSION_ID=\\"(.*)\\"', data)
            if version:
                debian_facts['distribution_version'] = version.group(1)
                debian_facts['distribution_major_version'] = version.group(1).split('.')[0]
        elif 'Deepin' in data or 'deepin' in data:
            debian_facts['distribution'] = 'Deepin'
            release = re.search('VERSION_CODENAME=\\"?([^\\"]+)\\"?', data)
            if release:
                debian_facts['distribution_release'] = release.groups()[0]
            version = re.search('VERSION_ID=\\"(.*)\\"', data)
            if version:
                debian_facts['distribution_version'] = version.group(1)
                debian_facts['distribution_major_version'] = version.group(1).split('.')[0]
        else:
            return (False, debian_facts)
        return (True, debian_facts)

    def parse_distribution_file_Mandriva(self, name, data, path, collected_facts):
        mandriva_facts = {}
        if 'Mandriva' in data:
            mandriva_facts['distribution'] = 'Mandriva'
            version = re.search('DISTRIB_RELEASE="(.*)"', data)
            if version:
                mandriva_facts['distribution_version'] = version.groups()[0]
            release = re.search('DISTRIB_CODENAME="(.*)"', data)
            if release:
                mandriva_facts['distribution_release'] = release.groups()[0]
            mandriva_facts['distribution'] = name
        else:
            return (False, mandriva_facts)
        return (True, mandriva_facts)

    def parse_distribution_file_NA(self, name, data, path, collected_facts):
        na_facts = {}
        for line in data.splitlines():
            distribution = re.search('^NAME=(.*)', line)
            if distribution and name == 'NA':
                na_facts['distribution'] = distribution.group(1).strip('"')
            version = re.search('^VERSION=(.*)', line)
            if version and collected_facts['distribution_version'] == 'NA':
                na_facts['distribution_version'] = version.group(1).strip('"')
        return (True, na_facts)

    def parse_distribution_file_Coreos(self, name, data, path, collected_facts):
        coreos_facts = {}
        distro = get_distribution()
        if distro.lower() == 'coreos':
            if not data:
                return (False, coreos_facts)
            release = re.search('^GROUP=(.*)', data)
            if release:
                coreos_facts['distribution_release'] = release.group(1).strip('"')
        else:
            return (False, coreos_facts)
        return (True, coreos_facts)

    def parse_distribution_file_Flatcar(self, name, data, path, collected_facts):
        flatcar_facts = {}
        distro = get_distribution()
        if distro.lower() != 'flatcar':
            return (False, flatcar_facts)
        if not data:
            return (False, flatcar_facts)
        version = re.search('VERSION=(.*)', data)
        if version:
            flatcar_facts['distribution_major_version'] = version.group(1).strip('"').split('.')[0]
            flatcar_facts['distribution_version'] = version.group(1).strip('"')
        return (True, flatcar_facts)

    def parse_distribution_file_ClearLinux(self, name, data, path, collected_facts):
        clear_facts = {}
        if 'clearlinux' not in name.lower():
            return (False, clear_facts)
        pname = re.search('NAME="(.*)"', data)
        if pname:
            if 'Clear Linux' not in pname.groups()[0]:
                return (False, clear_facts)
            clear_facts['distribution'] = pname.groups()[0]
        version = re.search('VERSION_ID=(.*)', data)
        if version:
            clear_facts['distribution_major_version'] = version.groups()[0]
            clear_facts['distribution_version'] = version.groups()[0]
        release = re.search('ID=(.*)', data)
        if release:
            clear_facts['distribution_release'] = release.groups()[0]
        return (True, clear_facts)

    def parse_distribution_file_CentOS(self, name, data, path, collected_facts):
        centos_facts = {}
        if 'CentOS Stream' in data:
            centos_facts['distribution_release'] = 'Stream'
            return (True, centos_facts)
        if 'TencentOS Server' in data:
            centos_facts['distribution'] = 'TencentOS'
            return (True, centos_facts)
        return (False, centos_facts)