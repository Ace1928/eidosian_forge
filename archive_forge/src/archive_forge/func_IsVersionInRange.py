from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import image_versions_util as image_version_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core.util import semver
def IsVersionInRange(version, range_from, range_to, loose=False):
    """Checks if given `version` is in range of (`range_from`, `range_to`).

  Args:
    version: version to check
    range_from: left boundary of range (inclusive), if None - no boundary
    range_to: right boundary of range (exclusive), if None - no boundary
    loose: if true use LooseVersion to compare, use SemVer otherwise

  Returns:
    True if given version is in range, otherwise False.
  """
    compare_fn = CompareLooseVersions if loose else CompareVersions
    return (range_from is None or compare_fn(range_from, version) <= 0) and (range_to is None or compare_fn(version, range_to) < 0)