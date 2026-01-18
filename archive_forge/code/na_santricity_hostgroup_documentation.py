from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
Apply desired host group state to the storage array.