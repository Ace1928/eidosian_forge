from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def _filter_clusters(self, return_keys, **kw):
    date = (datetime(2012, 10, 29, 13, 42, 2),)
    clusters = [{'id': '1', 'name': 'cluster1@lvmdriver-1', 'state': 'up', 'status': 'enabled', 'binary': 'cinder-volume', 'is_up': 'True', 'disabled': 'False', 'disabled_reason': None, 'num_hosts': '3', 'num_down_hosts': '2', 'updated_at': date, 'created_at': date, 'last_heartbeat': date}, {'id': '2', 'name': 'cluster1@lvmdriver-2', 'state': 'down', 'status': 'enabled', 'binary': 'cinder-volume', 'is_up': 'False', 'disabled': 'False', 'disabled_reason': None, 'num_hosts': '2', 'num_down_hosts': '2', 'updated_at': date, 'created_at': date, 'last_heartbeat': date}, {'id': '3', 'name': 'cluster2', 'state': 'up', 'status': 'disabled', 'binary': 'cinder-backup', 'is_up': 'True', 'disabled': 'True', 'disabled_reason': 'Reason', 'num_hosts': '1', 'num_down_hosts': '0', 'updated_at': date, 'created_at': date, 'last_heartbeat': date}]
    for key, value in kw.items():
        clusters = [cluster for cluster in clusters if cluster[key] == str(value)]
    result = []
    for cluster in clusters:
        result.append({key: cluster[key] for key in return_keys})
    return result