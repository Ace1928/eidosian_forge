from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def InsertEmptyFhirConfig(api_version):

    def VersionedInsertEmptyFhirConfig(flag):
        if not flag:
            return None
        messages = apis.GetMessagesModule('healthcare', api_version)
        return messages.FhirConfig()
    return VersionedInsertEmptyFhirConfig