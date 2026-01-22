import re
def _git_to_bzr_name(self, git_name):
    if git_name == b'master':
        bazaar_name = 'trunk'
    elif self._GIT_TRUNK_RE.match(git_name):
        bazaar_name = 'git-{}'.format(git_name.decode('utf-8'))
    else:
        bazaar_name = git_name.decode('utf-8')
    return bazaar_name