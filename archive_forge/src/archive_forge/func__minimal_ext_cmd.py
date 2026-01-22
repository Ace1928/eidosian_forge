import os
import subprocess
def _minimal_ext_cmd(cmd):
    env = {}
    for k in ['SYSTEMROOT', 'PATH']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=os.path.join(os.path.dirname(ROOT_DIR))) as proc:
        stdout, stderr = proc.communicate()
        if proc.returncode > 0:
            error_message = stderr.strip().decode('ascii')
            raise OSError(f'Command {cmd} exited with code {proc.returncode}: {error_message}')
    return stdout