from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
def what_provides(self, repoq, req_spec, qf=def_qf):
    if not repoq:
        pkgs = []
        try:
            try:
                pkgs = self.yum_base.returnPackagesByDep(req_spec) + self.yum_base.returnInstalledPackagesByDep(req_spec)
            except Exception as e:
                if 'repomd.xml signature could not be verified' in to_native(e):
                    if self.releasever:
                        self.module.run_command(self.yum_basecmd + ['makecache', 'fast', '--releasever=%s' % self.releasever])
                    else:
                        self.module.run_command(self.yum_basecmd + ['makecache', 'fast'])
                    pkgs = self.yum_base.returnPackagesByDep(req_spec) + self.yum_base.returnInstalledPackagesByDep(req_spec)
                else:
                    raise
            if not pkgs:
                exact_matches, glob_matches = self.yum_base.pkgSack.matchPackageNames([req_spec])[0:2]
                pkgs.extend(exact_matches)
                pkgs.extend(glob_matches)
                exact_matches, glob_matches = self.yum_base.rpmdb.matchPackageNames([req_spec])[0:2]
                pkgs.extend(exact_matches)
                pkgs.extend(glob_matches)
        except Exception as e:
            self.module.fail_json(msg='Failure talking to yum: %s' % to_native(e))
        return set((self.po_to_envra(p) for p in pkgs))
    else:
        myrepoq = list(repoq)
        r_cmd = ['--disablerepo', ','.join(self.disablerepo)]
        myrepoq.extend(r_cmd)
        r_cmd = ['--enablerepo', ','.join(self.enablerepo)]
        myrepoq.extend(r_cmd)
        if self.releasever:
            myrepoq.extend('--releasever=%s' % self.releasever)
        cmd = myrepoq + ['--qf', qf, '--whatprovides', req_spec]
        rc, out, err = self.module.run_command(cmd)
        cmd = myrepoq + ['--qf', qf, req_spec]
        rc2, out2, err2 = self.module.run_command(cmd)
        if rc == 0 and rc2 == 0:
            out += out2
            pkgs = {p for p in out.split('\n') if p.strip()}
            if not pkgs:
                pkgs = self.is_installed(repoq, req_spec, qf=qf)
            return pkgs
        else:
            self.module.fail_json(msg='Error from repoquery: %s: %s' % (cmd, err + err2))
    return set()