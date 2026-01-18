import os
import pathlib
import shutil
import nox
@nox.session(python=['2.7'])
def unit_prev_versions(session):
    constraints_path = str(CURRENT_DIRECTORY / 'testing' / f'constraints-{session.python}.txt')
    session.install('-r', 'testing/requirements.txt', '-c', constraints_path)
    session.install('-e', '.', '-c', constraints_path)
    session.run('pytest', f'--junitxml=unit_{session.python}_sponge_log.xml', '--cov=google.auth', '--cov=google.oauth2', '--cov=tests', '--ignore=tests/test_pluggable.py', 'tests', '--ignore=tests/transport/test__custom_tls_signer.py')