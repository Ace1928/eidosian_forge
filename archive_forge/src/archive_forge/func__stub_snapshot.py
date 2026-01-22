from cinderclient.tests.unit.fixture_data import base
def _stub_snapshot(**kwargs):
    snapshot = {'created_at': '2012-08-28T16:30:31.000000', 'display_description': None, 'display_name': None, 'id': '11111111-1111-1111-1111-111111111111', 'size': 1, 'status': 'available', 'volume_id': '00000000-0000-0000-0000-000000000000'}
    snapshot.update(kwargs)
    return snapshot