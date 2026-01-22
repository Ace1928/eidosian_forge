from __future__ import absolute_import, division, print_function
import copy
import glob
import json
import os
import re
import sys
import tempfile
import random
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.locale import get_best_parsable_locale
class SourcesList(object):

    def __init__(self, module):
        self.module = module
        self.files = {}
        self.new_repos = set()
        self.default_file = self._apt_cfg_file('Dir::Etc::sourcelist')
        if os.path.isfile(self.default_file):
            self.load(self.default_file)
        for file in glob.iglob('%s/*.list' % self._apt_cfg_dir('Dir::Etc::sourceparts')):
            self.load(file)

    def __iter__(self):
        """Simple iterator to go over all sources. Empty, non-source, and other not valid lines will be skipped."""
        for file, sources in self.files.items():
            for n, valid, enabled, source, comment in sources:
                if valid:
                    yield (file, n, enabled, source, comment)

    def _expand_path(self, filename):
        if '/' in filename:
            return filename
        else:
            return os.path.abspath(os.path.join(self._apt_cfg_dir('Dir::Etc::sourceparts'), filename))

    def _suggest_filename(self, line):

        def _cleanup_filename(s):
            filename = self.module.params['filename']
            if filename is not None:
                return filename
            return '_'.join(re.sub('[^a-zA-Z0-9]', ' ', s).split())

        def _strip_username_password(s):
            if '@' in s:
                s = s.split('@', 1)
                s = s[-1]
            return s
        line = re.sub('\\[[^\\]]+\\]', '', line)
        line = re.sub('\\w+://', '', line)
        parts = [part for part in line.split() if part not in VALID_SOURCE_TYPES]
        parts[0] = _strip_username_password(parts[0])
        return '%s.list' % _cleanup_filename(' '.join(parts[:1]))

    def _parse(self, line, raise_if_invalid_or_disabled=False):
        valid = False
        enabled = True
        source = ''
        comment = ''
        line = line.strip()
        if line.startswith('#'):
            enabled = False
            line = line[1:]
        i = line.find('#')
        if i > 0:
            comment = line[i + 1:].strip()
            line = line[:i]
        source = line.strip()
        if source:
            chunks = source.split()
            if chunks[0] in VALID_SOURCE_TYPES:
                valid = True
                source = ' '.join(chunks)
        if raise_if_invalid_or_disabled and (not valid or not enabled):
            raise InvalidSource(line)
        return (valid, enabled, source, comment)

    @staticmethod
    def _apt_cfg_file(filespec):
        """
        Wrapper for `apt_pkg` module for running with Python 2.5
        """
        try:
            result = apt_pkg.config.find_file(filespec)
        except AttributeError:
            result = apt_pkg.Config.FindFile(filespec)
        return result

    @staticmethod
    def _apt_cfg_dir(dirspec):
        """
        Wrapper for `apt_pkg` module for running with Python 2.5
        """
        try:
            result = apt_pkg.config.find_dir(dirspec)
        except AttributeError:
            result = apt_pkg.Config.FindDir(dirspec)
        return result

    def load(self, file):
        group = []
        f = open(file, 'r')
        for n, line in enumerate(f):
            valid, enabled, source, comment = self._parse(line)
            group.append((n, valid, enabled, source, comment))
        self.files[file] = group

    def save(self):
        for filename, sources in list(self.files.items()):
            if sources:
                d, fn = os.path.split(filename)
                try:
                    os.makedirs(d)
                except OSError as ex:
                    if not os.path.isdir(d):
                        self.module.fail_json('Failed to create directory %s: %s' % (d, to_native(ex)))
                try:
                    fd, tmp_path = tempfile.mkstemp(prefix='.%s-' % fn, dir=d)
                except (OSError, IOError) as e:
                    self.module.fail_json(msg='Unable to create temp file at "%s" for apt source: %s' % (d, to_native(e)))
                f = os.fdopen(fd, 'w')
                for n, valid, enabled, source, comment in sources:
                    chunks = []
                    if not enabled:
                        chunks.append('# ')
                    chunks.append(source)
                    if comment:
                        chunks.append(' # ')
                        chunks.append(comment)
                    chunks.append('\n')
                    line = ''.join(chunks)
                    try:
                        f.write(line)
                    except IOError as ex:
                        self.module.fail_json(msg='Failed to write to file %s: %s' % (tmp_path, to_native(ex)))
                self.module.atomic_move(tmp_path, filename)
                if filename in self.new_repos:
                    this_mode = self.module.params.get('mode', DEFAULT_SOURCES_PERM)
                    self.module.set_mode_if_different(filename, this_mode, False)
            else:
                del self.files[filename]
                if os.path.exists(filename):
                    os.remove(filename)

    def dump(self):
        dumpstruct = {}
        for filename, sources in self.files.items():
            if sources:
                lines = []
                for n, valid, enabled, source, comment in sources:
                    chunks = []
                    if not enabled:
                        chunks.append('# ')
                    chunks.append(source)
                    if comment:
                        chunks.append(' # ')
                        chunks.append(comment)
                    chunks.append('\n')
                    lines.append(''.join(chunks))
                dumpstruct[filename] = ''.join(lines)
        return dumpstruct

    def _choice(self, new, old):
        if new is None:
            return old
        return new

    def modify(self, file, n, enabled=None, source=None, comment=None):
        """
        This function to be used with iterator, so we don't care of invalid sources.
        If source, enabled, or comment is None, original value from line ``n`` will be preserved.
        """
        valid, enabled_old, source_old, comment_old = self.files[file][n][1:]
        self.files[file][n] = (n, valid, self._choice(enabled, enabled_old), self._choice(source, source_old), self._choice(comment, comment_old))

    def _add_valid_source(self, source_new, comment_new, file):
        self.module.log('ading source file: %s | %s | %s' % (source_new, comment_new, file))
        found = False
        for filename, n, enabled, source, comment in self:
            if source == source_new:
                self.modify(filename, n, enabled=True)
                found = True
        if not found:
            if file is None:
                file = self.default_file
            else:
                file = self._expand_path(file)
            if file not in self.files:
                self.files[file] = []
            files = self.files[file]
            files.append((len(files), True, True, source_new, comment_new))
            self.new_repos.add(file)

    def add_source(self, line, comment='', file=None):
        source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
        self._add_valid_source(source, comment, file=file or self._suggest_filename(source))

    def _remove_valid_source(self, source):
        for filename, n, enabled, src, comment in self:
            if source == src and enabled:
                self.files[filename].pop(n)

    def remove_source(self, line):
        source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
        self._remove_valid_source(source)