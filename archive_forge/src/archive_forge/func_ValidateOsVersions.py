from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def ValidateOsVersions(args, catalog_mgr):
    """Validate os-version-ids strings against the TestEnvironmentCatalog.

  Also allow users to alternatively specify OS version strings (e.g. '5.1.x')
  but translate them here to their corresponding version IDs (e.g. '22').
  The final list of validated version IDs is sorted in ascending order.

  Args:
    args: an argparse namespace. All the arguments that were provided to the
      command invocation (i.e. group and command arguments combined).
    catalog_mgr: an AndroidCatalogManager object for working with the Android
      TestEnvironmentCatalog.
  """
    if not args.os_version_ids:
        return
    validated_versions = set()
    for vers in args.os_version_ids:
        version_id = catalog_mgr.ValidateDimensionAndValue('version', vers)
        validated_versions.add(version_id)
    args.os_version_ids = sorted(validated_versions)