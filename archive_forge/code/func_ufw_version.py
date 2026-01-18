from __future__ import absolute_import, division, print_function
import re
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule
def ufw_version():
    """
        Returns the major and minor version of ufw installed on the system.
        """
    out = execute([[ufw_bin], ['--version']])
    lines = [x for x in out.split('\n') if x.strip() != '']
    if len(lines) == 0:
        module.fail_json(msg='Failed to get ufw version.', rc=0, out=out)
    matches = re.search('^ufw.+(\\d+)\\.(\\d+)(?:\\.(\\d+))?.*$', lines[0])
    if matches is None:
        module.fail_json(msg='Failed to get ufw version.', rc=0, out=out)
    major = int(matches.group(1))
    minor = int(matches.group(2))
    rev = 0
    if matches.group(3) is not None:
        rev = int(matches.group(3))
    return (major, minor, rev)