from __future__ import absolute_import, division, print_function
import errno
import os
import platform
import random
import re
import string
import filecmp
from ansible.module_utils.basic import AnsibleModule, get_distribution
from ansible.module_utils.six import iteritems
class BSDTimezone(Timezone):
    """This is the timezone implementation for *BSD which works simply through
    updating the `/etc/localtime` symlink to point to a valid timezone name under
    `/usr/share/zoneinfo`.
    """

    def __init__(self, module):
        super(BSDTimezone, self).__init__(module)

    def __get_timezone(self):
        zoneinfo_dir = '/usr/share/zoneinfo/'
        localtime_file = '/etc/localtime'
        if not os.path.exists(localtime_file):
            self.module.warn('Could not read /etc/localtime. Assuming UTC.')
            return 'UTC'
        zoneinfo_file = localtime_file
        while not zoneinfo_file.startswith(zoneinfo_dir):
            try:
                zoneinfo_file = os.readlink(localtime_file)
            except OSError:
                break
        else:
            return zoneinfo_file.replace(zoneinfo_dir, '')
        for dname, dummy, fnames in sorted(os.walk(zoneinfo_dir)):
            for fname in sorted(fnames):
                zoneinfo_file = os.path.join(dname, fname)
                if not os.path.islink(zoneinfo_file) and filecmp.cmp(zoneinfo_file, localtime_file):
                    return zoneinfo_file.replace(zoneinfo_dir, '')
        self.module.warn('Could not identify timezone name from /etc/localtime. Assuming UTC.')
        return 'UTC'

    def get(self, key, phase):
        """Lookup the current timezone by resolving `/etc/localtime`."""
        if key == 'name':
            return self.__get_timezone()
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)

    def set(self, key, value):
        if key == 'name':
            zonefile = '/usr/share/zoneinfo/' + value
            try:
                if not os.path.isfile(zonefile):
                    self.module.fail_json(msg='%s is not a recognized timezone' % value)
            except Exception:
                self.module.fail_json(msg='Failed to stat %s' % zonefile)
            suffix = ''.join([random.choice(string.ascii_letters + string.digits) for x in range(0, 10)])
            new_localtime = '/etc/localtime.' + suffix
            try:
                os.symlink(zonefile, new_localtime)
                os.rename(new_localtime, '/etc/localtime')
            except Exception:
                os.remove(new_localtime)
                self.module.fail_json(msg='Could not update /etc/localtime')
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)