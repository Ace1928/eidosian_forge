from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_cgsnapshot(detailed=True, **kwargs):
    cgsnapshot = {'name': None, 'id': '11111111-1111-1111-1111-111111111111'}
    if detailed:
        details = {'created_at': '2012-08-28T16:30:31.000000', 'description': None, 'name': None, 'id': '11111111-1111-1111-1111-111111111111', 'status': 'available', 'consistencygroup_id': '00000000-0000-0000-0000-000000000000'}
        cgsnapshot.update(details)
    cgsnapshot.update(kwargs)
    return cgsnapshot