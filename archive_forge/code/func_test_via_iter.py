from glance.tests import utils as test_utils
def test_via_iter(self):
    data = b''.join(list(test_utils.FakeData(1024)))
    self.assertEqual(1024, len(data))