from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
def _get_utilization(self, enclosure, params):
    fields = view = refresh = filter = ''
    if isinstance(params, dict):
        fields = params.get('fields')
        view = params.get('view')
        refresh = params.get('refresh')
        filter = params.get('filter')
    return self.oneview_client.enclosures.get_utilization(enclosure['uri'], fields=fields, filter=filter, refresh=refresh, view=view)