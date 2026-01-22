from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def _stub_group_snapshot(detailed=True, **kwargs):
    group_snapshot = {'name': None, 'id': '5678'}
    if detailed:
        details = {'created_at': '2012-08-28T16:30:31.000000', 'description': None, 'name': None, 'id': '5678', 'status': 'available', 'group_id': '1234'}
        group_snapshot.update(details)
    group_snapshot.update(kwargs)
    return group_snapshot