import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
def _create_trace(self, name, timestamp, parent_id='8d28af1e-acc0-498c-9890-6908e33eff5f', base_id=BASE_ID, trace_id='e465db5c-9672-45a1-b90b-da918f30aef6'):
    return {'parent_id': parent_id, 'name': name, 'base_id': base_id, 'trace_id': trace_id, 'timestamp': timestamp, 'info': {'host': self._host}}