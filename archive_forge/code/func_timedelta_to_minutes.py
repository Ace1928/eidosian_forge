from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def timedelta_to_minutes(time):
    if not time:
        return 0
    return time.days * 1440 + time.seconds / 60.0 + time.microseconds / 60000000.0