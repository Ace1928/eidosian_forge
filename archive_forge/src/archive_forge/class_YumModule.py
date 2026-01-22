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
class YumModule(YumDnf):
    """
    Yum Ansible module back-end implementation
    """

    def __init__(self, module):
        super(YumModule, self).__init__(module)
        self.pkg_mgr_name = 'yum'
        self.lockfile = '/var/run/yum.pid'
        self._yum_base = None

    def _enablerepos_with_error_checking(self):
        if len(self.enablerepo) == 1:
            try:
                self.yum_base.repos.enableRepo(self.enablerepo[0])
            except yum.Errors.YumBaseError as e:
                if u'repository not found' in to_text(e):
                    self.module.fail_json(msg='Repository %s not found.' % self.enablerepo[0])
                else:
                    raise e
        else:
            for rid in self.enablerepo:
                try:
                    self.yum_base.repos.enableRepo(rid)
                except yum.Errors.YumBaseError as e:
                    if u'repository not found' in to_text(e):
                        self.module.warn('Repository %s not found.' % rid)
                    else:
                        raise e

    def is_lockfile_pid_valid(self):
        try:
            try:
                with open(self.lockfile, 'r') as f:
                    oldpid = int(f.readline())
            except ValueError:
                os.unlink(self.lockfile)
                return False
            if oldpid == os.getpid():
                os.unlink(self.lockfile)
                return False
            try:
                with open('/proc/%d/stat' % oldpid, 'r') as f:
                    stat = f.readline()
                if stat.split()[2] == 'Z':
                    os.unlink(self.lockfile)
                    return False
            except IOError:
                try:
                    os.kill(oldpid, 0)
                except OSError as e:
                    if e.errno == errno.ESRCH:
                        os.unlink(self.lockfile)
                        return False
                    self.module.fail_json(msg='Unable to check PID %s in  %s: %s' % (oldpid, self.lockfile, to_native(e)))
        except (IOError, OSError) as e:
            return False
        return True

    @property
    def yum_base(self):
        if self._yum_base:
            return self._yum_base
        else:
            self._yum_base = yum.YumBase()
            self._yum_base.preconf.debuglevel = 0
            self._yum_base.preconf.errorlevel = 0
            self._yum_base.preconf.plugins = True
            self._yum_base.preconf.enabled_plugins = self.enable_plugin
            self._yum_base.preconf.disabled_plugins = self.disable_plugin
            if self.releasever:
                self._yum_base.preconf.releasever = self.releasever
            if self.installroot != '/':
                self._yum_base.preconf.root = self.installroot
                self._yum_base.conf.installroot = self.installroot
            if self.conf_file and os.path.exists(self.conf_file):
                self._yum_base.preconf.fn = self.conf_file
            if os.geteuid() != 0:
                if hasattr(self._yum_base, 'setCacheDir'):
                    self._yum_base.setCacheDir()
                else:
                    cachedir = yum.misc.getCacheDir()
                    self._yum_base.repos.setCacheDir(cachedir)
                    self._yum_base.conf.cache = 0
            if self.disable_excludes:
                self._yum_base.conf.disable_excludes = self.disable_excludes
            self._yum_base.conf.sslverify = self.sslverify
            self.yum_base.conf
            try:
                for rid in self.disablerepo:
                    self.yum_base.repos.disableRepo(rid)
                self._enablerepos_with_error_checking()
            except Exception as e:
                self.module.fail_json(msg='Failure talking to yum: %s' % to_native(e))
        return self._yum_base

    def po_to_envra(self, po):
        if hasattr(po, 'ui_envra'):
            return po.ui_envra
        return '%s:%s-%s-%s.%s' % (po.epoch, po.name, po.version, po.release, po.arch)

    def is_group_env_installed(self, name):
        name_lower = name.lower()
        if yum.__version_info__ >= (3, 4):
            groups_list = self.yum_base.doGroupLists(return_evgrps=True)
        else:
            groups_list = self.yum_base.doGroupLists()
        groups = groups_list[0]
        for group in groups:
            if name_lower.endswith(group.name.lower()) or name_lower.endswith(group.groupid.lower()):
                return True
        if yum.__version_info__ >= (3, 4):
            envs = groups_list[2]
            for env in envs:
                if name_lower.endswith(env.name.lower()) or name_lower.endswith(env.environmentid.lower()):
                    return True
        return False

    def is_installed(self, repoq, pkgspec, qf=None, is_pkg=False):
        if qf is None:
            qf = '%{epoch}:%{name}-%{version}-%{release}.%{arch}\n'
        if not repoq:
            pkgs = []
            try:
                e, m, dummy = self.yum_base.rpmdb.matchPackageNames([pkgspec])
                pkgs = e + m
                if not pkgs and (not is_pkg):
                    pkgs.extend(self.yum_base.returnInstalledPackagesByDep(pkgspec))
            except Exception as e:
                self.module.fail_json(msg='Failure talking to yum: %s' % to_native(e))
            return [self.po_to_envra(p) for p in pkgs]
        else:
            global rpmbin
            if not rpmbin:
                rpmbin = self.module.get_bin_path('rpm', required=True)
            cmd = [rpmbin, '-q', '--qf', qf, pkgspec]
            if '*' in pkgspec:
                cmd.append('-a')
            if self.installroot != '/':
                cmd.extend(['--root', self.installroot])
            locale = get_best_parsable_locale(self.module)
            lang_env = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale)
            rc, out, err = self.module.run_command(cmd, environ_update=lang_env)
            if rc != 0 and 'is not installed' not in out:
                self.module.fail_json(msg='Error from rpm: %s: %s' % (cmd, err))
            if 'is not installed' in out:
                out = ''
            pkgs = [p for p in out.replace('(none)', '0').split('\n') if p.strip()]
            if not pkgs and (not is_pkg):
                cmd = [rpmbin, '-q', '--qf', qf, '--whatprovides', pkgspec]
                if self.installroot != '/':
                    cmd.extend(['--root', self.installroot])
                rc2, out2, err2 = self.module.run_command(cmd, environ_update=lang_env)
            else:
                rc2, out2, err2 = (0, '', '')
            if rc2 != 0 and 'no package provides' not in out2:
                self.module.fail_json(msg='Error from rpm: %s: %s' % (cmd, err + err2))
            if 'no package provides' in out2:
                out2 = ''
            pkgs += [p for p in out2.replace('(none)', '0').split('\n') if p.strip()]
            return pkgs
        return []

    def is_available(self, repoq, pkgspec, qf=def_qf):
        if not repoq:
            pkgs = []
            try:
                e, m, dummy = self.yum_base.pkgSack.matchPackageNames([pkgspec])
                pkgs = e + m
                if not pkgs:
                    pkgs.extend(self.yum_base.returnPackagesByDep(pkgspec))
            except Exception as e:
                self.module.fail_json(msg='Failure talking to yum: %s' % to_native(e))
            return [self.po_to_envra(p) for p in pkgs]
        else:
            myrepoq = list(repoq)
            r_cmd = ['--disablerepo', ','.join(self.disablerepo)]
            myrepoq.extend(r_cmd)
            r_cmd = ['--enablerepo', ','.join(self.enablerepo)]
            myrepoq.extend(r_cmd)
            if self.releasever:
                myrepoq.extend('--releasever=%s' % self.releasever)
            cmd = myrepoq + ['--qf', qf, pkgspec]
            rc, out, err = self.module.run_command(cmd)
            if rc == 0:
                return [p for p in out.split('\n') if p.strip()]
            else:
                self.module.fail_json(msg='Error from repoquery: %s: %s' % (cmd, err))
        return []

    def is_update(self, repoq, pkgspec, qf=def_qf):
        if not repoq:
            pkgs = []
            updates = []
            try:
                pkgs = self.yum_base.returnPackagesByDep(pkgspec) + self.yum_base.returnInstalledPackagesByDep(pkgspec)
                if not pkgs:
                    e, m, dummy = self.yum_base.pkgSack.matchPackageNames([pkgspec])
                    pkgs = e + m
                updates = self.yum_base.doPackageLists(pkgnarrow='updates').updates
            except Exception as e:
                self.module.fail_json(msg='Failure talking to yum: %s' % to_native(e))
            retpkgs = (pkg for pkg in pkgs if pkg in updates)
            return set((self.po_to_envra(p) for p in retpkgs))
        else:
            myrepoq = list(repoq)
            r_cmd = ['--disablerepo', ','.join(self.disablerepo)]
            myrepoq.extend(r_cmd)
            r_cmd = ['--enablerepo', ','.join(self.enablerepo)]
            myrepoq.extend(r_cmd)
            if self.releasever:
                myrepoq.extend('--releasever=%s' % self.releasever)
            cmd = myrepoq + ['--pkgnarrow=updates', '--qf', qf, pkgspec]
            rc, out, err = self.module.run_command(cmd)
            if rc == 0:
                return set((p for p in out.split('\n') if p.strip()))
            else:
                self.module.fail_json(msg='Error from repoquery: %s: %s' % (cmd, err))
        return set()

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

    def transaction_exists(self, pkglist):
        """
        checks the package list to see if any packages are
        involved in an incomplete transaction
        """
        conflicts = []
        if not transaction_helpers:
            return conflicts
        pkglist_nvreas = (splitFilename(pkg) for pkg in pkglist)
        unfinished_transactions = find_unfinished_transactions()
        for trans in unfinished_transactions:
            steps = find_ts_remaining(trans)
            for step in steps:
                action, step_spec = step
                n, v, r, e, a = splitFilename(step_spec)
                for pkg in pkglist_nvreas:
                    label = '%s-%s' % (n, a)
                    if n == pkg[0] and a == pkg[4]:
                        if label not in conflicts:
                            conflicts.append('%s-%s' % (n, a))
                        break
        return conflicts

    def local_envra(self, path):
        """return envra of a local rpm passed in"""
        ts = rpm.TransactionSet()
        ts.setVSFlags(rpm._RPMVSF_NOSIGNATURES)
        fd = os.open(path, os.O_RDONLY)
        try:
            header = ts.hdrFromFdno(fd)
        except rpm.error as e:
            return None
        finally:
            os.close(fd)
        return '%s:%s-%s-%s.%s' % (header[rpm.RPMTAG_EPOCH] or '0', header[rpm.RPMTAG_NAME], header[rpm.RPMTAG_VERSION], header[rpm.RPMTAG_RELEASE], header[rpm.RPMTAG_ARCH])

    @contextmanager
    def set_env_proxy(self):
        namepass = ''
        scheme = ['http', 'https']
        old_proxy_env = [os.getenv('http_proxy'), os.getenv('https_proxy')]
        try:
            if self.yum_base.conf.proxy and self.yum_base.conf.proxy not in ('_none_',):
                if self.yum_base.conf.proxy_username:
                    namepass = namepass + self.yum_base.conf.proxy_username
                    proxy_url = self.yum_base.conf.proxy
                    if self.yum_base.conf.proxy_password:
                        namepass = namepass + ':' + self.yum_base.conf.proxy_password
                elif '@' in self.yum_base.conf.proxy:
                    namepass = self.yum_base.conf.proxy.split('@')[0].split('//')[-1]
                    proxy_url = self.yum_base.conf.proxy.replace('{0}@'.format(namepass), '')
                if namepass:
                    namepass = namepass + '@'
                    for item in scheme:
                        os.environ[item + '_proxy'] = re.sub('(http://)', '\\g<1>' + namepass, proxy_url)
                else:
                    for item in scheme:
                        os.environ[item + '_proxy'] = self.yum_base.conf.proxy
            yield
        except yum.Errors.YumBaseError:
            raise
        finally:
            for item in scheme:
                if os.getenv('{0}_proxy'.format(item)):
                    del os.environ['{0}_proxy'.format(item)]
            if old_proxy_env[0]:
                os.environ['http_proxy'] = old_proxy_env[0]
            if old_proxy_env[1]:
                os.environ['https_proxy'] = old_proxy_env[1]

    def pkg_to_dict(self, pkgstr):
        if pkgstr.strip() and pkgstr.count('|') == 5:
            n, e, v, r, a, repo = pkgstr.split('|')
        else:
            return {'error_parsing': pkgstr}
        d = {'name': n, 'arch': a, 'epoch': e, 'release': r, 'version': v, 'repo': repo, 'envra': '%s:%s-%s-%s.%s' % (e, n, v, r, a)}
        if repo == 'installed':
            d['yumstate'] = 'installed'
        else:
            d['yumstate'] = 'available'
        return d

    def repolist(self, repoq, qf='%{repoid}'):
        cmd = repoq + ['--qf', qf, '-a']
        if self.releasever:
            cmd.extend(['--releasever=%s' % self.releasever])
        rc, out, err = self.module.run_command(cmd)
        if rc == 0:
            return set((p for p in out.split('\n') if p.strip()))
        else:
            return []

    def list_stuff(self, repoquerybin, stuff):
        qf = '%{name}|%{epoch}|%{version}|%{release}|%{arch}|%{repoid}'
        is_installed_qf = '%{name}|%{epoch}|%{version}|%{release}|%{arch}|installed\n'
        repoq = [repoquerybin, '--show-duplicates', '--plugins', '--quiet']
        if self.disablerepo:
            repoq.extend(['--disablerepo', ','.join(self.disablerepo)])
        if self.enablerepo:
            repoq.extend(['--enablerepo', ','.join(self.enablerepo)])
        if self.installroot != '/':
            repoq.extend(['--installroot', self.installroot])
        if self.conf_file and os.path.exists(self.conf_file):
            repoq += ['-c', self.conf_file]
        if stuff == 'installed':
            return [self.pkg_to_dict(p) for p in sorted(self.is_installed(repoq, '-a', qf=is_installed_qf)) if p.strip()]
        if stuff == 'updates':
            return [self.pkg_to_dict(p) for p in sorted(self.is_update(repoq, '-a', qf=qf)) if p.strip()]
        if stuff == 'available':
            return [self.pkg_to_dict(p) for p in sorted(self.is_available(repoq, '-a', qf=qf)) if p.strip()]
        if stuff == 'repos':
            return [dict(repoid=name, state='enabled') for name in sorted(self.repolist(repoq)) if name.strip()]
        return [self.pkg_to_dict(p) for p in sorted(self.is_installed(repoq, stuff, qf=is_installed_qf) + self.is_available(repoq, stuff, qf=qf)) if p.strip()]

    def exec_install(self, items, action, pkgs, res):
        cmd = self.yum_basecmd + [action] + pkgs
        if self.releasever:
            cmd.extend(['--releasever=%s' % self.releasever])
        if not self.sslverify:
            cmd.extend(['--setopt', 'sslverify=0'])
        if self.module.check_mode:
            self.module.exit_json(changed=True, results=res['results'], changes=dict(installed=pkgs))
        else:
            res['changes'] = dict(installed=pkgs)
        locale = get_best_parsable_locale(self.module)
        lang_env = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale)
        rc, out, err = self.module.run_command(cmd, environ_update=lang_env)
        if rc == 1:
            for spec in items:
                if '://' in spec and ('No package %s available.' % spec in out or 'Cannot open: %s. Skipping.' % spec in err):
                    err = 'Package at %s could not be installed' % spec
                    self.module.fail_json(changed=False, msg=err, rc=rc)
        res['rc'] = rc
        res['results'].append(out)
        res['msg'] += err
        res['changed'] = True
        if 'Nothing to do' in out and rc == 0 or 'does not have any packages' in err:
            res['changed'] = False
        if rc != 0:
            res['changed'] = False
            self.module.fail_json(**res)
        if 'No space left on device' in (out or err):
            res['changed'] = False
            res['msg'] = 'No space left on device'
            self.module.fail_json(**res)
        return res

    def install(self, items, repoq):
        pkgs = []
        downgrade_pkgs = []
        res = {}
        res['results'] = []
        res['msg'] = ''
        res['rc'] = 0
        res['changed'] = False
        for spec in items:
            pkg = None
            downgrade_candidate = False
            if spec.endswith('.rpm') or '://' in spec:
                if '://' not in spec and (not os.path.exists(spec)):
                    res['msg'] += "No RPM file matching '%s' found on system" % spec
                    res['results'].append("No RPM file matching '%s' found on system" % spec)
                    res['rc'] = 127
                    self.module.fail_json(**res)
                if '://' in spec:
                    with self.set_env_proxy():
                        package = fetch_file(self.module, spec)
                        if not package.endswith('.rpm'):
                            new_package_path = '%s.rpm' % package
                            os.rename(package, new_package_path)
                            package = new_package_path
                else:
                    package = spec
                envra = self.local_envra(package)
                if envra is None:
                    self.module.fail_json(msg='Failed to get envra information from RPM package: %s' % spec)
                installed_pkgs = self.is_installed(repoq, envra)
                if installed_pkgs:
                    res['results'].append('%s providing %s is already installed' % (installed_pkgs[0], package))
                    continue
                name, ver, rel, epoch, arch = splitFilename(envra)
                installed_pkgs = self.is_installed(repoq, name)
                if len(installed_pkgs) == 2:
                    cur_name0, cur_ver0, cur_rel0, cur_epoch0, cur_arch0 = splitFilename(installed_pkgs[0])
                    cur_name1, cur_ver1, cur_rel1, cur_epoch1, cur_arch1 = splitFilename(installed_pkgs[1])
                    cur_epoch0 = cur_epoch0 or '0'
                    cur_epoch1 = cur_epoch1 or '0'
                    compare = compareEVR((cur_epoch0, cur_ver0, cur_rel0), (cur_epoch1, cur_ver1, cur_rel1))
                    if compare == 0 and cur_arch0 != cur_arch1:
                        for installed_pkg in installed_pkgs:
                            if installed_pkg.endswith(arch):
                                installed_pkgs = [installed_pkg]
                if len(installed_pkgs) == 1:
                    installed_pkg = installed_pkgs[0]
                    cur_name, cur_ver, cur_rel, cur_epoch, cur_arch = splitFilename(installed_pkg)
                    cur_epoch = cur_epoch or '0'
                    compare = compareEVR((cur_epoch, cur_ver, cur_rel), (epoch, ver, rel))
                    if compare > 0 and self.allow_downgrade:
                        downgrade_candidate = True
                    elif compare >= 0:
                        continue
                pkg = package
            elif spec.startswith('@'):
                if self.is_group_env_installed(spec):
                    continue
                pkg = spec
            else:
                if not set(['*', '?']).intersection(set(spec)):
                    installed_pkgs = self.is_installed(repoq, spec, is_pkg=True)
                    if installed_pkgs:
                        res['results'].append('%s providing %s is already installed' % (installed_pkgs[0], spec))
                        continue
                pkglist = self.what_provides(repoq, spec)
                if not pkglist:
                    res['msg'] += "No package matching '%s' found available, installed or updated" % spec
                    res['results'].append("No package matching '%s' found available, installed or updated" % spec)
                    res['rc'] = 126
                    self.module.fail_json(**res)
                conflicts = self.transaction_exists(pkglist)
                if conflicts:
                    res['msg'] += 'The following packages have pending transactions: %s' % ', '.join(conflicts)
                    res['rc'] = 125
                    self.module.fail_json(**res)
                found = False
                for this in pkglist:
                    if self.is_installed(repoq, this, is_pkg=True):
                        found = True
                        res['results'].append('%s providing %s is already installed' % (this, spec))
                        break
                if not found:
                    if self.is_installed(repoq, spec):
                        found = True
                        res['results'].append('package providing %s is already installed' % spec)
                if found:
                    continue
                if self.allow_downgrade:
                    for package in pkglist:
                        name, ver, rel, epoch, arch = splitFilename(package)
                        inst_pkgs = self.is_installed(repoq, name, is_pkg=True)
                        if inst_pkgs:
                            cur_name, cur_ver, cur_rel, cur_epoch, cur_arch = splitFilename(inst_pkgs[0])
                            compare = compareEVR((cur_epoch, cur_ver, cur_rel), (epoch, ver, rel))
                            if compare > 0:
                                downgrade_candidate = True
                            else:
                                downgrade_candidate = False
                                break
                pkg = spec
            if downgrade_candidate and self.allow_downgrade:
                downgrade_pkgs.append(pkg)
            else:
                pkgs.append(pkg)
        if downgrade_pkgs:
            res = self.exec_install(items, 'downgrade', downgrade_pkgs, res)
        if pkgs:
            res = self.exec_install(items, 'install', pkgs, res)
        return res

    def remove(self, items, repoq):
        pkgs = []
        res = {}
        res['results'] = []
        res['msg'] = ''
        res['changed'] = False
        res['rc'] = 0
        for pkg in items:
            if pkg.startswith('@'):
                installed = self.is_group_env_installed(pkg)
            else:
                installed = self.is_installed(repoq, pkg)
            if installed:
                pkgs.append(pkg)
            else:
                res['results'].append('%s is not installed' % pkg)
        if pkgs:
            if self.module.check_mode:
                self.module.exit_json(changed=True, results=res['results'], changes=dict(removed=pkgs))
            else:
                res['changes'] = dict(removed=pkgs)
            if self.autoremove:
                cmd = self.yum_basecmd + ['autoremove'] + pkgs
            else:
                cmd = self.yum_basecmd + ['remove'] + pkgs
            rc, out, err = self.module.run_command(cmd)
            res['rc'] = rc
            res['results'].append(out)
            res['msg'] = err
            if rc != 0:
                if self.autoremove and 'No such command' in out:
                    self.module.fail_json(msg='Version of YUM too old for autoremove: Requires yum 3.4.3 (RHEL/CentOS 7+)')
                else:
                    self.module.fail_json(**res)
            self._yum_base = None
            for pkg in pkgs:
                if pkg.startswith('@'):
                    installed = self.is_group_env_installed(pkg)
                else:
                    installed = self.is_installed(repoq, pkg, is_pkg=True)
                if installed:
                    res['msg'] = "Package '%s' couldn't be removed!" % pkg
                    self.module.fail_json(**res)
            res['changed'] = True
        return res

    def run_check_update(self):
        if self.releasever:
            rc, out, err = self.module.run_command(self.yum_basecmd + ['check-update'] + ['--releasever=%s' % self.releasever])
        else:
            rc, out, err = self.module.run_command(self.yum_basecmd + ['check-update'])
        return (rc, out, err)

    @staticmethod
    def parse_check_update(check_update_output):
        out = '\n'.join((l for l in check_update_output.splitlines() if l))
        out = re.sub('\\n\\W+(.*)', ' \\1', out)
        updates = {}
        obsoletes = {}
        for line in out.split('\n'):
            line = line.split()
            if '*' in line or len(line) not in [3, 6] or '.' not in line[0]:
                continue
            pkg, version, repo = (line[0], line[1], line[2])
            name, dist = pkg.rsplit('.', 1)
            if name not in updates:
                updates[name] = []
            updates[name].append({'version': version, 'dist': dist, 'repo': repo})
            if len(line) == 6:
                obsolete_pkg, obsolete_version, obsolete_repo = (line[3], line[4], line[5])
                obsolete_name, obsolete_dist = obsolete_pkg.rsplit('.', 1)
                if obsolete_name not in obsoletes:
                    obsoletes[obsolete_name] = []
                obsoletes[obsolete_name].append({'version': obsolete_version, 'dist': obsolete_dist, 'repo': obsolete_repo})
        return (updates, obsoletes)

    def latest(self, items, repoq):
        res = {}
        res['results'] = []
        res['msg'] = ''
        res['changed'] = False
        res['rc'] = 0
        pkgs = {}
        pkgs['update'] = []
        pkgs['install'] = []
        updates = {}
        obsoletes = {}
        update_all = False
        cmd = self.yum_basecmd[:]
        if '*' in items:
            update_all = True
        rc, out, err = self.run_check_update()
        if rc == 0 and update_all:
            res['results'].append('Nothing to do here, all packages are up to date')
            return res
        elif rc == 100:
            updates, obsoletes = self.parse_check_update(out)
        elif rc == 1:
            res['msg'] = err
            res['rc'] = rc
            self.module.fail_json(**res)
        if update_all:
            cmd.append('update')
            will_update = set(updates.keys())
            will_update_from_other_package = dict()
        else:
            will_update = set()
            will_update_from_other_package = dict()
            for spec in items:
                if spec.startswith('@'):
                    pkgs['update'].append(spec)
                    will_update.add(spec)
                    continue
                if spec.endswith('.rpm') and '://' not in spec:
                    if not os.path.exists(spec):
                        res['msg'] += "No RPM file matching '%s' found on system" % spec
                        res['results'].append("No RPM file matching '%s' found on system" % spec)
                        res['rc'] = 127
                        self.module.fail_json(**res)
                    envra = self.local_envra(spec)
                    if envra is None:
                        self.module.fail_json(msg='Failed to get envra information from RPM package: %s' % spec)
                    if self.is_installed(repoq, envra):
                        pkgs['update'].append(spec)
                    else:
                        pkgs['install'].append(spec)
                    continue
                if '://' in spec:
                    with self.set_env_proxy():
                        package = fetch_file(self.module, spec)
                    envra = self.local_envra(package)
                    if envra is None:
                        self.module.fail_json(msg='Failed to get envra information from RPM package: %s' % spec)
                    if self.is_installed(repoq, envra):
                        pkgs['update'].append(spec)
                    else:
                        pkgs['install'].append(spec)
                    continue
                if self.is_installed(repoq, spec):
                    pkgs['update'].append(spec)
                else:
                    pkgs['install'].append(spec)
                pkglist = self.what_provides(repoq, spec)
                if not pkglist:
                    res['msg'] += "No package matching '%s' found available, installed or updated" % spec
                    res['results'].append("No package matching '%s' found available, installed or updated" % spec)
                    res['rc'] = 126
                    self.module.fail_json(**res)
                nothing_to_do = True
                for pkg in pkglist:
                    if spec in pkgs['install'] and self.is_available(repoq, pkg):
                        nothing_to_do = False
                        break
                    pkgname, ver, rel, epoch, arch = splitFilename(pkg)
                    if spec in pkgs['update'] and pkgname in updates:
                        nothing_to_do = False
                        will_update.add(spec)
                        if spec != pkgname:
                            will_update_from_other_package[spec] = pkgname
                        break
                if not self.is_installed(repoq, spec) and self.update_only:
                    res['results'].append('Packages providing %s not installed due to update_only specified' % spec)
                    continue
                if nothing_to_do:
                    res['results'].append('All packages providing %s are up to date' % spec)
                    continue
                conflicts = self.transaction_exists(pkglist)
                if conflicts:
                    res['msg'] += 'The following packages have pending transactions: %s' % ', '.join(conflicts)
                    res['results'].append('The following packages have pending transactions: %s' % ', '.join(conflicts))
                    res['rc'] = 128
                    self.module.fail_json(**res)
        to_update = []
        for w in will_update:
            if w.startswith('@'):
                to_update.append((w, None))
            elif w not in updates:
                other_pkg = will_update_from_other_package[w]
                pkg = updates[other_pkg][0]
                to_update.append((w, 'because of (at least) %s-%s.%s from %s' % (other_pkg, pkg['version'], pkg['dist'], pkg['repo'])))
            else:
                for pkg in updates[w]:
                    to_update.append((w, '%s.%s from %s' % (pkg['version'], pkg['dist'], pkg['repo'])))
        if self.update_only:
            res['changes'] = dict(installed=[], updated=to_update)
        else:
            res['changes'] = dict(installed=pkgs['install'], updated=to_update)
        if obsoletes:
            res['obsoletes'] = obsoletes
        if self.module.check_mode:
            if will_update or pkgs['install']:
                res['changed'] = True
            return res
        if self.releasever:
            cmd.extend(['--releasever=%s' % self.releasever])
        if update_all:
            rc, out, err = self.module.run_command(cmd)
            res['changed'] = True
        elif self.update_only:
            if pkgs['update']:
                cmd += ['update'] + pkgs['update']
                locale = get_best_parsable_locale(self.module)
                lang_env = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale)
                rc, out, err = self.module.run_command(cmd, environ_update=lang_env)
                out_lower = out.strip().lower()
                if not out_lower.endswith('no packages marked for update') and (not out_lower.endswith('nothing to do')):
                    res['changed'] = True
            else:
                rc, out, err = [0, '', '']
        elif pkgs['install'] or (will_update and (not self.update_only)):
            cmd += ['install'] + pkgs['install'] + pkgs['update']
            locale = get_best_parsable_locale(self.module)
            lang_env = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale)
            rc, out, err = self.module.run_command(cmd, environ_update=lang_env)
            out_lower = out.strip().lower()
            if not out_lower.endswith('no packages marked for update') and (not out_lower.endswith('nothing to do')):
                res['changed'] = True
        else:
            rc, out, err = [0, '', '']
        res['rc'] = rc
        res['msg'] += err
        res['results'].append(out)
        if rc:
            res['failed'] = True
        return res

    def ensure(self, repoq):
        pkgs = self.names
        if not self.names and self.autoremove:
            pkgs = []
            self.state = 'absent'
        if self.conf_file and os.path.exists(self.conf_file):
            self.yum_basecmd += ['-c', self.conf_file]
            if repoq:
                repoq += ['-c', self.conf_file]
        if self.skip_broken:
            self.yum_basecmd.extend(['--skip-broken'])
        if self.disablerepo:
            self.yum_basecmd.extend(['--disablerepo=%s' % ','.join(self.disablerepo)])
        if self.enablerepo:
            self.yum_basecmd.extend(['--enablerepo=%s' % ','.join(self.enablerepo)])
        if self.enable_plugin:
            self.yum_basecmd.extend(['--enableplugin', ','.join(self.enable_plugin)])
        if self.disable_plugin:
            self.yum_basecmd.extend(['--disableplugin', ','.join(self.disable_plugin)])
        if self.exclude:
            e_cmd = ['--exclude=%s' % ','.join(self.exclude)]
            self.yum_basecmd.extend(e_cmd)
        if self.disable_excludes:
            self.yum_basecmd.extend(['--disableexcludes=%s' % self.disable_excludes])
        if self.cacheonly:
            self.yum_basecmd.extend(['--cacheonly'])
        if self.download_only:
            self.yum_basecmd.extend(['--downloadonly'])
            if self.download_dir:
                self.yum_basecmd.extend(['--downloaddir=%s' % self.download_dir])
        if self.releasever:
            self.yum_basecmd.extend(['--releasever=%s' % self.releasever])
        if self.installroot != '/':
            e_cmd = ['--installroot=%s' % self.installroot]
            self.yum_basecmd.extend(e_cmd)
        if self.state in ('installed', 'present', 'latest'):
            if self.update_cache:
                self.module.run_command(self.yum_basecmd + ['clean', 'expire-cache'])
            try:
                current_repos = self.yum_base.repos.repos.keys()
                if self.enablerepo:
                    try:
                        new_repos = self.yum_base.repos.repos.keys()
                        for i in new_repos:
                            if i not in current_repos:
                                rid = self.yum_base.repos.getRepo(i)
                                a = rid.repoXML.repoid
                        current_repos = new_repos
                    except yum.Errors.YumBaseError as e:
                        self.module.fail_json(msg='Error setting/accessing repos: %s' % to_native(e))
            except yum.Errors.YumBaseError as e:
                self.module.fail_json(msg='Error accessing repos: %s' % to_native(e))
        if self.state == 'latest' or self.update_only:
            if self.disable_gpg_check:
                self.yum_basecmd.append('--nogpgcheck')
            if self.security:
                self.yum_basecmd.append('--security')
            if self.bugfix:
                self.yum_basecmd.append('--bugfix')
            res = self.latest(pkgs, repoq)
        elif self.state in ('installed', 'present'):
            if self.disable_gpg_check:
                self.yum_basecmd.append('--nogpgcheck')
            res = self.install(pkgs, repoq)
        elif self.state in ('removed', 'absent'):
            res = self.remove(pkgs, repoq)
        else:
            self.module.fail_json(msg='we should never get here unless this all failed', changed=False, results='', errors='unexpected state')
        return res

    @staticmethod
    def has_yum():
        return HAS_YUM_PYTHON

    def run(self):
        """
        actually execute the module code backend
        """
        if (not HAS_RPM_PYTHON or not HAS_YUM_PYTHON) and sys.executable != '/usr/bin/python' and (not has_respawned()):
            respawn_module('/usr/bin/python')
        error_msgs = []
        if not HAS_RPM_PYTHON:
            error_msgs.append('The Python 2 bindings for rpm are needed for this module. If you require Python 3 support use the `dnf` Ansible module instead.')
        if not HAS_YUM_PYTHON:
            error_msgs.append('The Python 2 yum module is needed for this module. If you require Python 3 support use the `dnf` Ansible module instead.')
        self.wait_for_lock()
        if error_msgs:
            self.module.fail_json(msg='. '.join(error_msgs))
        if self.module.get_bin_path('yum-deprecated'):
            yumbin = self.module.get_bin_path('yum-deprecated')
        else:
            yumbin = self.module.get_bin_path('yum')
        self.yum_basecmd = [yumbin, '-d', '2', '-y']
        if self.update_cache and (not self.names) and (not self.list):
            rc, stdout, stderr = self.module.run_command(self.yum_basecmd + ['clean', 'expire-cache'])
            if rc == 0:
                self.module.exit_json(changed=False, msg='Cache updated', rc=rc, results=[])
            else:
                self.module.exit_json(changed=False, msg='Failed to update cache', rc=rc, results=[stderr])
        repoquerybin = self.module.get_bin_path('repoquery', required=False)
        if self.install_repoquery and (not repoquerybin) and (not self.module.check_mode):
            yum_path = self.module.get_bin_path('yum')
            if yum_path:
                if self.releasever:
                    self.module.run_command('%s -y install yum-utils --releasever %s' % (yum_path, self.releasever))
                else:
                    self.module.run_command('%s -y install yum-utils' % yum_path)
            repoquerybin = self.module.get_bin_path('repoquery', required=False)
        if self.list:
            if not repoquerybin:
                self.module.fail_json(msg='repoquery is required to use list= with this module. Please install the yum-utils package.')
            results = {'results': self.list_stuff(repoquerybin, self.list)}
        else:
            repoquery = None
            try:
                yum_plugins = self.yum_base.plugins._plugins
            except AttributeError:
                pass
            else:
                if 'rhnplugin' in yum_plugins:
                    if repoquerybin:
                        repoquery = [repoquerybin, '--show-duplicates', '--plugins', '--quiet']
                        if self.installroot != '/':
                            repoquery.extend(['--installroot', self.installroot])
                        if self.disable_excludes:
                            try:
                                with open('/etc/yum.conf', 'r') as f:
                                    content = f.readlines()
                                tmp_conf_file = tempfile.NamedTemporaryFile(dir=self.module.tmpdir, delete=False)
                                self.module.add_cleanup_file(tmp_conf_file.name)
                                tmp_conf_file.writelines([c for c in content if not c.startswith('exclude=')])
                                tmp_conf_file.close()
                            except Exception as e:
                                self.module.fail_json(msg='Failure setting up repoquery: %s' % to_native(e))
                            repoquery.extend(['-c', tmp_conf_file.name])
            results = self.ensure(repoquery)
            if repoquery:
                results['msg'] = '%s %s' % (results.get('msg', ''), 'Warning: Due to potential bad behaviour with rhnplugin and certificates, used slower repoquery calls instead of Yum API.')
        self.module.exit_json(**results)