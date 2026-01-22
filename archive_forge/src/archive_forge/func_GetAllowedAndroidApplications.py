from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base as calliope_base
def GetAllowedAndroidApplications(args, messages):
    """Create list of allowed android applications."""
    allowed_applications = []
    for application in getattr(args, 'allowed_application', []) or []:
        android_application = messages.V2AndroidApplication(sha1Fingerprint=application['sha1_fingerprint'], packageName=application['package_name'])
        allowed_applications.append(android_application)
    return allowed_applications