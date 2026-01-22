from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def _refresh_status():
    try:
        client.get(url)
    except HwcClientException404:
        return (True, 'Done')
    except Exception:
        return (None, '')
    return (True, 'Pending')