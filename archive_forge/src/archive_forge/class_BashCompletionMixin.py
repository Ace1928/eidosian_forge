import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
class BashCompletionMixin:
    """Component for testing execution of a bash completion script."""
    _test_needs_features = [features.bash_feature]
    script = None

    def complete(self, words, cword=-1):
        """Perform a bash completion.

        :param words: a list of words representing the current command.
        :param cword: the current word to complete, defaults to the last one.
        """
        if self.script is None:
            self.script = self.get_script()
        env = dict(os.environ)
        env['PYTHONPATH'] = ':'.join(sys.path)
        proc = subprocess.Popen([features.bash_feature.path, '--noprofile'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if cword < 0:
            cword = len(words) + cword
        encoding = osutils.get_user_encoding()
        input = b'%s\n' % self.script.encode(encoding)
        input += b'COMP_WORDS=( %s )\n' % b' '.join([b"'" + w.replace("'", "'\\''").encode(encoding) + b"'" for w in words])
        input += b'COMP_CWORD=%d\n' % cword
        input += b'%s\n' % getattr(self, 'script_name', '_brz').encode(encoding)
        input += b'echo ${#COMPREPLY[*]}\n'
        input += b"IFS=$'\\n'\n"
        input += b'echo "${COMPREPLY[*]}"\n'
        out, err = proc.communicate(input)
        errlines = [line for line in err.splitlines() if not line.startswith(b'brz: warning: ')]
        if [] != errlines:
            raise AssertionError('Unexpected error message:\n%s' % err)
        self.assertEqual(b'', b''.join(errlines), 'No messages to standard error')
        lines = out.split(b'\n')
        nlines = int(lines[0])
        del lines[0]
        self.assertEqual(b'', lines[-1], 'Newline at end')
        del lines[-1]
        if nlines == 0 and len(lines) == 1 and (lines[0] == b''):
            del lines[0]
        self.assertEqual(nlines, len(lines), 'No newlines in generated words')
        self.completion_result = {l.decode(encoding) for l in lines}
        return self.completion_result

    def assertCompletionEquals(self, *words):
        self.assertEqual(set(words), self.completion_result)

    def assertCompletionContains(self, *words):
        missing = set(words) - self.completion_result
        if missing:
            raise AssertionError('Completion should contain %r but it has %r' % (missing, self.completion_result))

    def assertCompletionOmits(self, *words):
        surplus = set(words) & self.completion_result
        if surplus:
            raise AssertionError('Completion should omit %r but it has %r' % (surplus, self.completion_result))

    def get_script(self):
        commands.install_bzr_command_hooks()
        dc = DataCollector()
        data = dc.collect()
        cg = BashCodeGen(data)
        res = cg.function()
        return res