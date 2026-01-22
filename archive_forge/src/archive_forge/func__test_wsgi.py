import os
import re
import subprocess
import sys
from pbr.tests import base
def _test_wsgi(self, cmd_name, output, extra_args=None):
    cmd = os.path.join(self.temp_dir, 'bin', cmd_name)
    print('Running %s -p 0 -b 127.0.0.1' % cmd)
    popen_cmd = [cmd, '-p', '0', '-b', '127.0.0.1']
    if extra_args:
        popen_cmd.extend(extra_args)
    env = {'PYTHONPATH': self._get_path()}
    p = subprocess.Popen(popen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.temp_dir, env=env)
    self.addCleanup(p.kill)
    stdoutdata = p.stdout.readline()
    stdoutdata = p.stdout.readline()
    self.assertIn(b'STARTING test server pbr_testpackage.wsgi', stdoutdata)
    stdoutdata = p.stdout.readline()
    print(stdoutdata)
    m = re.search(b'(http://[^:]+:\\d+)/', stdoutdata)
    self.assertIsNotNone(m, 'Regex failed to match on %s' % stdoutdata)
    stdoutdata = p.stdout.readline()
    self.assertIn(b'DANGER! For testing only, do not use in production', stdoutdata)
    stdoutdata = p.stdout.readline()
    f = urlopen(m.group(1).decode('utf-8'))
    self.assertEqual(output, f.read())
    urlopen(m.group(1).decode('utf-8'))
    stdoutdata = p.stderr.readline()
    status = '"GET / HTTP/1.1" 200 %d' % len(output)
    self.assertIn(status.encode('utf-8'), stdoutdata)