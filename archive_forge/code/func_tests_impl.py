import os
import shutil
import subprocess
import nox
def tests_impl(session, extras='socks,secure,brotli'):
    session.install('-r', 'dev-requirements.txt')
    session.install('.[{extras}]'.format(extras=extras))
    session.run('pip', '--version')
    session.run('python', '--version')
    session.run('python', '-c', "import struct; print(struct.calcsize('P') * 8)")
    session.run('python', '-m', 'OpenSSL.debug')
    session.run('coverage', 'run', '--parallel-mode', '-m', 'pytest', '-r', 'a', f'--color={('yes' if 'GITHUB_ACTIONS' in os.environ else 'auto')}', '--tb=native', '--no-success-flaky-report', *(session.posargs or ('test/',)), env={'PYTHONWARNINGS': 'always::DeprecationWarning'})