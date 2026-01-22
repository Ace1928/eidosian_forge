from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ZypperSettings(_messages.Message):
    """Zypper patching is performed by running `zypper patch`. See also
  https://en.opensuse.org/SDB:Zypper_manual.

  Fields:
    categories: Install only patches with these categories. Common categories
      include security, recommended, and feature.
    excludes: List of patches to exclude from update.
    exclusivePatches: An exclusive list of patches to be updated. These are
      the only patches that will be installed using 'zypper patch patch:'
      command. This field must not be used with any other patch configuration
      fields.
    severities: Install only patches with these severities. Common severities
      include critical, important, moderate, and low.
    withOptional: Adds the `--with-optional` flag to `zypper patch`.
    withUpdate: Adds the `--with-update` flag, to `zypper patch`.
  """
    categories = _messages.StringField(1, repeated=True)
    excludes = _messages.StringField(2, repeated=True)
    exclusivePatches = _messages.StringField(3, repeated=True)
    severities = _messages.StringField(4, repeated=True)
    withOptional = _messages.BooleanField(5)
    withUpdate = _messages.BooleanField(6)