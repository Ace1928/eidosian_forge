from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class YumSettings(_messages.Message):
    """Yum patching is performed by executing `yum update`. Additional options
  can be set to control how this is executed. Note that not all settings are
  supported on all platforms.

  Fields:
    excludes: List of packages to exclude from update. These packages are
      excluded by using the yum `--exclude` flag.
    exclusivePackages: An exclusive list of packages to be updated. These are
      the only packages that will be updated. If these packages are not
      installed, they will be ignored. This field must not be specified with
      any other patch configuration fields.
    minimal: Will cause patch to run `yum update-minimal` instead.
    security: Adds the `--security` flag to `yum update`. Not supported on all
      platforms.
  """
    excludes = _messages.StringField(1, repeated=True)
    exclusivePackages = _messages.StringField(2, repeated=True)
    minimal = _messages.BooleanField(3)
    security = _messages.BooleanField(4)