from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
def _TransformAppProfileToIsolationMode(app_profile):
    """Extracts the isolation mode from the app profile."""
    if 'dataBoostIsolationReadOnly' in app_profile:
        return 'DATA_BOOST_ISOLATION_READ_ONLY'
    return 'STANDARD_ISOLATION'