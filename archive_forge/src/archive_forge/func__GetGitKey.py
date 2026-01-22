from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetGitKey(obj):
    """Hash the Git specification for the given RepoSync|RootSync object."""
    repo = obj['spec']['git']['repo']
    branch = 'main'
    if 'branch' in obj['spec']['git']:
        branch = obj['spec']['git']['branch']
    directory = '.'
    if 'dir' in obj['spec']['git']:
        directory = obj['spec']['git']['dir']
    revision = ''
    if 'revision' in obj['spec']['git']:
        revision = obj['spec']['git']['revision']
    if not revision:
        return '{repo}//{dir}@{branch}'.format(repo=repo, dir=directory, branch=branch)
    else:
        return '{repo}//{dir}@{branch}:{revision}'.format(repo=repo, dir=directory, branch=branch, revision=revision)