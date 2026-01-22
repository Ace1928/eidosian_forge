from __future__ import absolute_import, division, print_function
import binascii
import codecs
import datetime
import fnmatch
import grp
import os
import platform
import pwd
import re
import stat
import time
import traceback
from functools import partial
from zipfile import ZipFile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_file
class TgzArchive(object):

    def __init__(self, src, b_dest, file_args, module):
        self.src = src
        self.b_dest = b_dest
        self.file_args = file_args
        self.opts = module.params['extra_opts']
        self.module = module
        if self.module.check_mode:
            self.module.exit_json(skipped=True, msg='remote module (%s) does not support check mode when using gtar' % self.module._name)
        self.excludes = [path.rstrip('/') for path in self.module.params['exclude']]
        self.include_files = self.module.params['include']
        self.cmd_path = None
        self.tar_type = None
        self.zipflag = '-z'
        self._files_in_archive = []

    def _get_tar_type(self):
        cmd = [self.cmd_path, '--version']
        rc, out, err = self.module.run_command(cmd)
        tar_type = None
        if out.startswith('bsdtar'):
            tar_type = 'bsd'
        elif out.startswith('tar') and 'GNU' in out:
            tar_type = 'gnu'
        return tar_type

    @property
    def files_in_archive(self):
        if self._files_in_archive:
            return self._files_in_archive
        cmd = [self.cmd_path, '--list', '-C', self.b_dest]
        if self.zipflag:
            cmd.append(self.zipflag)
        if self.opts:
            cmd.extend(['--show-transformed-names'] + self.opts)
        if self.excludes:
            cmd.extend(['--exclude=' + f for f in self.excludes])
        cmd.extend(['-f', self.src])
        if self.include_files:
            cmd.extend(self.include_files)
        locale = get_best_parsable_locale(self.module)
        rc, out, err = self.module.run_command(cmd, cwd=self.b_dest, environ_update=dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LANGUAGE=locale))
        if rc != 0:
            self.module.debug(err)
            raise UnarchiveError('Unable to list files in the archive: %s' % err)
        for filename in out.splitlines():
            filename = to_native(codecs.escape_decode(filename)[0])
            if filename.startswith('/'):
                filename = filename[1:]
            exclude_flag = False
            if self.excludes:
                for exclude in self.excludes:
                    if fnmatch.fnmatch(filename, exclude):
                        exclude_flag = True
                        break
            if not exclude_flag:
                self._files_in_archive.append(to_native(filename))
        return self._files_in_archive

    def is_unarchived(self):
        cmd = [self.cmd_path, '--diff', '-C', self.b_dest]
        if self.zipflag:
            cmd.append(self.zipflag)
        if self.opts:
            cmd.extend(['--show-transformed-names'] + self.opts)
        if self.file_args['owner']:
            cmd.append('--owner=' + quote(self.file_args['owner']))
        if self.file_args['group']:
            cmd.append('--group=' + quote(self.file_args['group']))
        if self.module.params['keep_newer']:
            cmd.append('--keep-newer-files')
        if self.excludes:
            cmd.extend(['--exclude=' + f for f in self.excludes])
        cmd.extend(['-f', self.src])
        if self.include_files:
            cmd.extend(self.include_files)
        locale = get_best_parsable_locale(self.module)
        rc, out, err = self.module.run_command(cmd, cwd=self.b_dest, environ_update=dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LANGUAGE=locale))
        unarchived = True
        old_out = out
        out = ''
        run_uid = os.getuid()
        for line in old_out.splitlines() + err.splitlines():
            if EMPTY_FILE_RE.search(line):
                continue
            if run_uid == 0 and (not self.file_args['owner']) and OWNER_DIFF_RE.search(line):
                out += line + '\n'
            if run_uid == 0 and (not self.file_args['group']) and GROUP_DIFF_RE.search(line):
                out += line + '\n'
            if not self.file_args['mode'] and MODE_DIFF_RE.search(line):
                out += line + '\n'
            if MOD_TIME_DIFF_RE.search(line):
                out += line + '\n'
            if MISSING_FILE_RE.search(line):
                out += line + '\n'
            if INVALID_OWNER_RE.search(line):
                out += line + '\n'
            if INVALID_GROUP_RE.search(line):
                out += line + '\n'
        if out:
            unarchived = False
        return dict(unarchived=unarchived, rc=rc, out=out, err=err, cmd=cmd)

    def unarchive(self):
        cmd = [self.cmd_path, '--extract', '-C', self.b_dest]
        if self.zipflag:
            cmd.append(self.zipflag)
        if self.opts:
            cmd.extend(['--show-transformed-names'] + self.opts)
        if self.file_args['owner']:
            cmd.append('--owner=' + quote(self.file_args['owner']))
        if self.file_args['group']:
            cmd.append('--group=' + quote(self.file_args['group']))
        if self.module.params['keep_newer']:
            cmd.append('--keep-newer-files')
        if self.excludes:
            cmd.extend(['--exclude=' + f for f in self.excludes])
        cmd.extend(['-f', self.src])
        if self.include_files:
            cmd.extend(self.include_files)
        locale = get_best_parsable_locale(self.module)
        rc, out, err = self.module.run_command(cmd, cwd=self.b_dest, environ_update=dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LANGUAGE=locale))
        return dict(cmd=cmd, rc=rc, out=out, err=err)

    def can_handle_archive(self):
        try:
            self.cmd_path = get_bin_path('gtar')
        except ValueError:
            try:
                self.cmd_path = get_bin_path('tar')
            except ValueError:
                return (False, "Unable to find required 'gtar' or 'tar' binary in the path")
        self.tar_type = self._get_tar_type()
        if self.tar_type != 'gnu':
            return (False, 'Command "%s" detected as tar type %s. GNU tar required.' % (self.cmd_path, self.tar_type))
        try:
            if self.files_in_archive:
                return (True, None)
        except UnarchiveError as e:
            return (False, 'Command "%s" could not handle archive: %s' % (self.cmd_path, to_native(e)))
        return (False, 'Command "%s" found no files in archive. Empty archive files are not supported.' % self.cmd_path)