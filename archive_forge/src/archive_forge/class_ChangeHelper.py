from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
class ChangeHelper:

    def __init__(self, old, new):
        self.key = new.key
        self.old = old
        self.new = new

    def __eq__(self, other):
        return (self.key, self.new.enabled, self.new.level) == (other.key, other.new.enabled, other.new.level)

    def __gt__(self, other):
        if self.key < other.key:
            if self.new.enabled < other.old.enabled:
                return True
            elif self.new.enabled > other.old.enabled:
                return False
            else:
                return self.new.level < other.old.level
        else:
            return not other > self

    def __ge__(self, other):
        return self > other or self == other

    def __lt__(self, other):
        return not self >= other

    def __le__(self, other):
        return not self > other