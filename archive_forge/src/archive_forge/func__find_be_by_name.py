from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def _find_be_by_name(self, out):
    if '@' in self.name:
        for line in out.splitlines():
            if self.is_freebsd:
                check = line.split()
                if check == []:
                    continue
                full_name = check[0].split('/')
                if full_name == []:
                    continue
                check[0] = full_name[len(full_name) - 1]
                if check[0] == self.name:
                    return check
            else:
                check = line.split(';')
                if check[0] == self.name:
                    return check
    else:
        for line in out.splitlines():
            if self.is_freebsd:
                check = line.split()
                if check[0] == self.name:
                    return check
            else:
                check = line.split(';')
                if check[0] == self.name:
                    return check
    return None