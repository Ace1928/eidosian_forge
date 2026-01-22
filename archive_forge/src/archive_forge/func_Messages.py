from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def Messages(api_version):
    return apis.GetMessagesModule('krmapihosting', api_version)