from __future__ import absolute_import, division, print_function
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
class XenServerFacts:

    def __init__(self):
        self.codes = {'5.5.0': 'george', '5.6.100': 'oxford', '6.0.0': 'boston', '6.1.0': 'tampa', '6.2.0': 'clearwater'}

    @property
    def version(self):
        result = distro.linux_distribution()[1]
        return result

    @property
    def codename(self):
        if self.version in self.codes:
            result = self.codes[self.version]
        else:
            result = None
        return result