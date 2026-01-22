from __future__ import absolute_import, division, print_function
import ctypes.util
import grp
import calendar
import os
import re
import pty
import pwd
import select
import shutil
import socket
import subprocess
import time
import math
from ansible.module_utils import distro
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
import ansible.module_utils.compat.typing as t
class SunOS(User):
    """
    This is a SunOS User manipulation class - The main difference between
    this class and the generic user class is that Solaris-type distros
    don't support the concept of a "system" account and we need to
    edit the /etc/shadow file manually to set a password. (Ugh)

    This overrides the following methods from the generic class:-
      - create_user()
      - remove_user()
      - modify_user()
      - user_info()
    """
    platform = 'SunOS'
    distribution = None
    SHADOWFILE = '/etc/shadow'
    USER_ATTR = '/etc/user_attr'

    def get_password_defaults(self):
        try:
            minweeks = ''
            maxweeks = ''
            warnweeks = ''
            with open('/etc/default/passwd', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or line == '':
                        continue
                    m = re.match('^([^#]*)#(.*)$', line)
                    if m:
                        line = m.group(1)
                    key, value = line.split('=')
                    if key == 'MINWEEKS':
                        minweeks = value.rstrip('\n')
                    elif key == 'MAXWEEKS':
                        maxweeks = value.rstrip('\n')
                    elif key == 'WARNWEEKS':
                        warnweeks = value.rstrip('\n')
        except Exception as err:
            self.module.fail_json(msg='failed to read /etc/default/passwd: %s' % to_native(err))
        return (minweeks, maxweeks, warnweeks)

    def remove_user(self):
        cmd = [self.module.get_bin_path('userdel', True)]
        if self.remove:
            cmd.append('-r')
        cmd.append(self.name)
        return self.execute_command(cmd)

    def create_user(self):
        cmd = [self.module.get_bin_path('useradd', True)]
        if self.uid is not None:
            cmd.append('-u')
            cmd.append(self.uid)
            if self.non_unique:
                cmd.append('-o')
        if self.group is not None:
            if not self.group_exists(self.group):
                self.module.fail_json(msg='Group %s does not exist' % self.group)
            cmd.append('-g')
            cmd.append(self.group)
        if self.groups is not None:
            groups = self.get_groups_set()
            cmd.append('-G')
            cmd.append(','.join(groups))
        if self.comment is not None:
            cmd.append('-c')
            cmd.append(self.comment)
        if self.home is not None:
            cmd.append('-d')
            cmd.append(self.home)
        if self.shell is not None:
            cmd.append('-s')
            cmd.append(self.shell)
        if self.create_home:
            cmd.append('-m')
            if self.skeleton is not None:
                cmd.append('-k')
                cmd.append(self.skeleton)
            if self.umask is not None:
                cmd.append('-K')
                cmd.append('UMASK=' + self.umask)
        if self.profile is not None:
            cmd.append('-P')
            cmd.append(self.profile)
        if self.authorization is not None:
            cmd.append('-A')
            cmd.append(self.authorization)
        if self.role is not None:
            cmd.append('-R')
            cmd.append(self.role)
        cmd.append(self.name)
        rc, out, err = self.execute_command(cmd)
        if rc is not None and rc != 0:
            self.module.fail_json(name=self.name, msg=err, rc=rc)
        if not self.module.check_mode:
            if self.password is not None:
                self.backup_shadow()
                minweeks, maxweeks, warnweeks = self.get_password_defaults()
                try:
                    lines = []
                    with open(self.SHADOWFILE, 'rb') as f:
                        for line in f:
                            line = to_native(line, errors='surrogate_or_strict')
                            fields = line.strip().split(':')
                            if not fields[0] == self.name:
                                lines.append(line)
                                continue
                            fields[1] = self.password
                            fields[2] = str(int(time.time() // 86400))
                            if minweeks:
                                try:
                                    fields[3] = str(int(minweeks) * 7)
                                except ValueError:
                                    pass
                            if maxweeks:
                                try:
                                    fields[4] = str(int(maxweeks) * 7)
                                except ValueError:
                                    pass
                            if warnweeks:
                                try:
                                    fields[5] = str(int(warnweeks) * 7)
                                except ValueError:
                                    pass
                            line = ':'.join(fields)
                            lines.append('%s\n' % line)
                    with open(self.SHADOWFILE, 'w+') as f:
                        f.writelines(lines)
                except Exception as err:
                    self.module.fail_json(msg='failed to update users password: %s' % to_native(err))
        return (rc, out, err)

    def modify_user_usermod(self):
        cmd = [self.module.get_bin_path('usermod', True)]
        cmd_len = len(cmd)
        info = self.user_info()
        if self.uid is not None and info[2] != int(self.uid):
            cmd.append('-u')
            cmd.append(self.uid)
            if self.non_unique:
                cmd.append('-o')
        if self.group is not None:
            if not self.group_exists(self.group):
                self.module.fail_json(msg='Group %s does not exist' % self.group)
            ginfo = self.group_info(self.group)
            if info[3] != ginfo[2]:
                cmd.append('-g')
                cmd.append(self.group)
        if self.groups is not None:
            current_groups = self.user_group_membership()
            groups = self.get_groups_set(names_only=True)
            group_diff = set(current_groups).symmetric_difference(groups)
            groups_need_mod = False
            if group_diff:
                if self.append:
                    for g in groups:
                        if g in group_diff:
                            groups_need_mod = True
                            break
                else:
                    groups_need_mod = True
            if groups_need_mod:
                cmd.append('-G')
                new_groups = groups
                if self.append:
                    new_groups.update(current_groups)
                cmd.append(','.join(new_groups))
        if self.comment is not None and info[4] != self.comment:
            cmd.append('-c')
            cmd.append(self.comment)
        if self.home is not None and info[5] != self.home:
            if self.move_home:
                cmd.append('-m')
            cmd.append('-d')
            cmd.append(self.home)
        if self.shell is not None and info[6] != self.shell:
            cmd.append('-s')
            cmd.append(self.shell)
        if self.profile is not None and info[7] != self.profile:
            cmd.append('-P')
            cmd.append(self.profile)
        if self.authorization is not None and info[8] != self.authorization:
            cmd.append('-A')
            cmd.append(self.authorization)
        if self.role is not None and info[9] != self.role:
            cmd.append('-R')
            cmd.append(self.role)
        if cmd_len != len(cmd):
            cmd.append(self.name)
            rc, out, err = self.execute_command(cmd)
            if rc is not None and rc != 0:
                self.module.fail_json(name=self.name, msg=err, rc=rc)
        else:
            rc, out, err = (None, '', '')
        if self.update_password == 'always' and self.password is not None and (info[1] != self.password):
            self.backup_shadow()
            rc, out, err = (0, '', '')
            if not self.module.check_mode:
                minweeks, maxweeks, warnweeks = self.get_password_defaults()
                try:
                    lines = []
                    with open(self.SHADOWFILE, 'rb') as f:
                        for line in f:
                            line = to_native(line, errors='surrogate_or_strict')
                            fields = line.strip().split(':')
                            if not fields[0] == self.name:
                                lines.append(line)
                                continue
                            fields[1] = self.password
                            fields[2] = str(int(time.time() // 86400))
                            if minweeks:
                                fields[3] = str(int(minweeks) * 7)
                            if maxweeks:
                                fields[4] = str(int(maxweeks) * 7)
                            if warnweeks:
                                fields[5] = str(int(warnweeks) * 7)
                            line = ':'.join(fields)
                            lines.append('%s\n' % line)
                    with open(self.SHADOWFILE, 'w+') as f:
                        f.writelines(lines)
                    rc = 0
                except Exception as err:
                    self.module.fail_json(msg='failed to update users password: %s' % to_native(err))
        return (rc, out, err)

    def user_info(self):
        info = super(SunOS, self).user_info()
        if info:
            info += self._user_attr_info()
        return info

    def _user_attr_info(self):
        info = [''] * 3
        with open(self.USER_ATTR, 'r') as file_handler:
            for line in file_handler:
                lines = line.strip().split('::::')
                if lines[0] == self.name:
                    tmp = dict((x.split('=') for x in lines[1].split(';')))
                    info[0] = tmp.get('profiles', '')
                    info[1] = tmp.get('auths', '')
                    info[2] = tmp.get('roles', '')
        return info