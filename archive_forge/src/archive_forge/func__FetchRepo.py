from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
from dulwich import client
from dulwich import errors
from dulwich import index
from dulwich import porcelain
from dulwich import repo
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
def _FetchRepo(target_dir, url):
    """Fetch a git repository from 'url' into 'target_dir'.

  See InstallRuntimeDef() for information on which version is selected.

  Args:
    target_dir: (str) Directory name.
    url: (str) Git repository URL.

  Raises:
    errors.HangupException: Hangup during communication to a remote repository.
  """
    if os.path.exists(target_dir):
        log.debug('Fetching from %s into existing directory.', url)
        try:
            porcelain.fetch(target_dir, url)
        except (IOError, OSError) as ex:
            raise InvalidTargetDirectoryError('Unable to fetch into target directory {0}: {1}'.format(target_dir, ex.message))
    else:
        try:
            log.debug('Cloning from %s into %s', url, target_dir)
            porcelain.clone(url, target_dir, checkout=False)
        except (errors.NotGitRepository, OSError) as ex:
            raise InvalidTargetDirectoryError('Unable to clone into target directory {0}: {1}'.format(target_dir, ex.message))
        except KeyError as ex:
            if ex.message == 'HEAD':
                raise InvalidRepositoryError()
            else:
                raise