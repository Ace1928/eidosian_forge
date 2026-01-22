from breezy.lazy_import import lazy_import
from ... import config, merge
import fnmatch
import subprocess
import tempfile
from breezy import (
class PoMerger(merge.PerFileMerger):
    """Merge .po files."""

    def __init__(self, merger):
        super(merge.PerFileMerger, self).__init__(merger)
        self.conf = merger.this_branch.get_config_stack()
        self.po_dirs = self.conf.get('po_merge.po_dirs')
        self.po_glob = self.conf.get('po_merge.po_glob')
        self.pot_glob = self.conf.get('po_merge.pot_glob')
        self.command = self.conf.get('po_merge.command', expand=False)
        self.pot_file_abspath = None
        trace.mutter('PoMerger created')

    def file_matches(self, params):
        """Return True if merge_matching should be called on this file."""
        if not self.po_dirs or not self.command:
            return False
        po_dir = None
        po_path = params.this_path
        for po_dir in self.po_dirs:
            glob = osutils.pathjoin(po_dir, self.po_glob)
            if fnmatch.fnmatch(po_path, glob):
                trace.mutter('po {} matches: {}'.format(po_path, glob))
                break
        else:
            trace.mutter('PoMerger did not match for %s and %s' % (self.po_dirs, self.po_glob))
            return False
        for path, file_class, kind, entry in self.merger.this_tree.list_files(from_dir=po_dir, recursive=False):
            pot_name, pot_file_id = (path, entry)
            if fnmatch.fnmatch(pot_name, self.pot_glob):
                relpath = osutils.pathjoin(po_dir, pot_name)
                self.pot_file_abspath = self.merger.this_tree.abspath(relpath)
                trace.mutter('will msgmerge %s using %s' % (po_path, self.pot_file_abspath))
                return True
        else:
            return False

    def _invoke(self, command):
        trace.mutter('Will msgmerge: {}'.format(command))
        proc = subprocess.Popen(cmdline.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        out, err = proc.communicate()
        return (proc.returncode, out, err)

    def merge_matching(self, params):
        return self.merge_text(params)

    def merge_text(self, params):
        """Calls msgmerge when .po files conflict.

        This requires a valid .pot file to reconcile both sides.
        """
        tmpdir = tempfile.mkdtemp(prefix='po_merge')
        env = {}
        env['this'] = osutils.pathjoin(tmpdir, 'this')
        env['other'] = osutils.pathjoin(tmpdir, 'other')
        env['result'] = osutils.pathjoin(tmpdir, 'result')
        env['pot_file'] = self.pot_file_abspath
        try:
            with open(env['this'], 'wb') as f:
                f.writelines(params.this_lines)
            with open(env['other'], 'wb') as f:
                f.writelines(params.other_lines)
            command = self.conf.expand_options(self.command, env)
            retcode, out, err = self._invoke(command)
            with open(env['result'], 'rb') as f:
                return ('success', list(f.readlines()))
        finally:
            osutils.rmtree(tmpdir)
        return ('not applicable', [])