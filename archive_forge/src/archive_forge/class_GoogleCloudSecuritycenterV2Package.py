from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Package(_messages.Message):
    """Package is a generic definition of a package.

  Fields:
    cpeUri: The CPE URI where the vulnerability was detected.
    packageName: The name of the package where the vulnerability was detected.
    packageType: Type of package, for example, os, maven, or go.
    packageVersion: The version of the package.
  """
    cpeUri = _messages.StringField(1)
    packageName = _messages.StringField(2)
    packageType = _messages.StringField(3)
    packageVersion = _messages.StringField(4)