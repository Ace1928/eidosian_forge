from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Detail(_messages.Message):
    """A detail for a distro and package affected by this vulnerability and its
  associated fix (if one is available).

  Fields:
    affectedCpeUri: Required. The [CPE
      URI](https://cpe.mitre.org/specification/) this vulnerability affects.
    affectedPackage: Required. The package this vulnerability affects.
    affectedVersionEnd: The version number at the end of an interval in which
      this vulnerability exists. A vulnerability can affect a package between
      version numbers that are disjoint sets of intervals (example:
      [1.0.0-1.1.0], [2.4.6-2.4.8] and [4.5.6-4.6.8]) each of which will be
      represented in its own Detail. If a specific affected version is
      provided by a vulnerability database, affected_version_start and
      affected_version_end will be the same in that Detail.
    affectedVersionStart: The version number at the start of an interval in
      which this vulnerability exists. A vulnerability can affect a package
      between version numbers that are disjoint sets of intervals (example:
      [1.0.0-1.1.0], [2.4.6-2.4.8] and [4.5.6-4.6.8]) each of which will be
      represented in its own Detail. If a specific affected version is
      provided by a vulnerability database, affected_version_start and
      affected_version_end will be the same in that Detail.
    description: A vendor-specific description of this vulnerability.
    fixedCpeUri: The distro recommended [CPE
      URI](https://cpe.mitre.org/specification/) to update to that contains a
      fix for this vulnerability. It is possible for this to be different from
      the affected_cpe_uri.
    fixedPackage: The distro recommended package to update to that contains a
      fix for this vulnerability. It is possible for this to be different from
      the affected_package.
    fixedVersion: The distro recommended version to update to that contains a
      fix for this vulnerability. Setting this to VersionKind.MAXIMUM means no
      such version is yet available.
    isObsolete: Whether this detail is obsolete. Occurrences are expected not
      to point to obsolete details.
    packageType: The type of package; whether native or non native (e.g., ruby
      gems, node.js packages, etc.).
    severityName: The distro assigned severity of this vulnerability.
    source: The source from which the information in this Detail was obtained.
    sourceUpdateTime: The time this information was last changed at the
      source. This is an upstream timestamp from the underlying information
      source - e.g. Ubuntu security tracker.
    vendor: The name of the vendor of the product.
  """
    affectedCpeUri = _messages.StringField(1)
    affectedPackage = _messages.StringField(2)
    affectedVersionEnd = _messages.MessageField('Version', 3)
    affectedVersionStart = _messages.MessageField('Version', 4)
    description = _messages.StringField(5)
    fixedCpeUri = _messages.StringField(6)
    fixedPackage = _messages.StringField(7)
    fixedVersion = _messages.MessageField('Version', 8)
    isObsolete = _messages.BooleanField(9)
    packageType = _messages.StringField(10)
    severityName = _messages.StringField(11)
    source = _messages.StringField(12)
    sourceUpdateTime = _messages.StringField(13)
    vendor = _messages.StringField(14)