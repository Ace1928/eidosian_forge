from __future__ import absolute_import
import nox
@nox.session
@nox.parametrize('py', ['3.5', '3.6'])
def tests(session, py):
    session.install('mock', 'pytest', 'pytest-cov')
    session.install('-e', '.[oauth2client]')
    session.run('py.test', '--quiet', '--cov=google_reauth', '--cov-config=.coveragerc', 'tests', *session.posargs)