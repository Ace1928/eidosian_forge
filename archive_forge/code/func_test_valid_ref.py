import testtools
import uuid
import barbicanclient
from barbicanclient import base
from barbicanclient import version
def test_valid_ref(self):
    secret_uuid = uuid.uuid4()
    ref = 'http://localhost/' + str(secret_uuid)
    self.assertEqual(secret_uuid, base.validate_ref_and_return_uuid(ref, 'Thing'))