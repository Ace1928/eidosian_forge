from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def a_valid_tap(tap):
    """Returns True if the tap is valid."""
    regex = re.compile('^([\\w-]+)/(homebrew-)?([\\w-]+)$')
    return regex.match(tap)