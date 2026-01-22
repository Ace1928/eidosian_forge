from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AptPackageConfig(_messages.Message):
    """A list of packages to install, remove, and their repos for a given
  package manager type.

  Fields:
    packageInstalls: Packages to install. apt-get update && apt-get -y install
      package1 package2 package3
    packageRemovals: Packages to remove. apt-get -y remove package1 package2
      package3
    repositories: Package repositories to configure in the package manager.
      The instance likely already has some defaults set and duplicates are
      acceptable but ignored.
  """
    packageInstalls = _messages.MessageField('Package', 1, repeated=True)
    packageRemovals = _messages.MessageField('Package', 2, repeated=True)
    repositories = _messages.MessageField('AptRepository', 3, repeated=True)