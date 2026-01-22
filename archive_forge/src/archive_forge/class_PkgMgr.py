from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common._utils import get_all_subclasses
class PkgMgr(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def is_available(self):
        pass

    @abstractmethod
    def list_installed(self):
        pass

    @abstractmethod
    def get_package_details(self, package):
        pass

    def get_packages(self):
        installed_packages = {}
        for package in self.list_installed():
            package_details = self.get_package_details(package)
            if 'source' not in package_details:
                package_details['source'] = self.__class__.__name__.lower()
            name = package_details['name']
            if name not in installed_packages:
                installed_packages[name] = [package_details]
            else:
                installed_packages[name].append(package_details)
        return installed_packages