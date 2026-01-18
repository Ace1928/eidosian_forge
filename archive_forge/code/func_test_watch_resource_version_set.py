import unittest
from mock import Mock, call
from .watch import Watch
def test_watch_resource_version_set(self):
    fake_resp = Mock()
    fake_resp.close = Mock()
    fake_resp.release_conn = Mock()
    values = ['{"type": "ADDED", "object": {"metadata": {"name": "test1","resourceVersion": "1"}, "spec": {}, "status": {}}}\n', '{"type": "ADDED", "object": {"metadata": {"name": "test2","resourceVersion": "2"}, "spec": {}, "sta', 'tus": {}}}\n{"type": "ADDED", "object": {"metadata": {"name": "test3","resourceVersion": "3"}, "spec": {}, "status": {}}}\n']

    def get_values(*args, **kwargs):
        self.callcount += 1
        if self.callcount == 1:
            return []
        else:
            return values
    fake_resp.read_chunked = Mock(side_effect=get_values)
    fake_api = Mock()
    fake_api.get_namespaces = Mock(return_value=fake_resp)
    fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
    w = Watch()
    calls = []
    iterations = 2
    calls.append(call(_preload_content=False, watch=True, resource_version='5'))
    calls.append(call(_preload_content=False, watch=True, resource_version='5'))
    for i in range(iterations):
        calls.append(call(_preload_content=False, watch=True, resource_version='3'))
    for c, e in enumerate(w.stream(fake_api.get_namespaces, resource_version='5')):
        if c == len(values) * iterations:
            w.stop()
    fake_api.get_namespaces.assert_has_calls(calls)
    self.assertEqual(fake_api.get_namespaces.mock_calls, calls)