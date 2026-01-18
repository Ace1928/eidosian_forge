import os
import subprocess
def write_version_py(filename='cvxpy/version.py'):
    cnt = "\n# THIS FILE IS GENERATED FROM CVXPY SETUP.PY\nshort_version = '%(version)s'\nversion = '%(version)s'\nfull_version = '%(full_version)s'\ngit_revision = '%(git_revision)s'\ncommit_count = '%(commit_count)s'\nrelease = %(isrelease)s\nif not release:\n    version = full_version\n"
    FULLVERSION, GIT_REVISION, COMMIT_COUNT = get_version_info()
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'full_version': FULLVERSION, 'git_revision': GIT_REVISION, 'commit_count': COMMIT_COUNT, 'isrelease': str(IS_RELEASED)})
    finally:
        a.close()