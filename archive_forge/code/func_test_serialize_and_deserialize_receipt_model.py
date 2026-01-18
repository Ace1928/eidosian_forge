import datetime
from unittest import mock
import uuid
from keystone.common.cache import _context_cache
from keystone.common import utils as ks_utils
from keystone import exception
from keystone.models import receipt_model
from keystone.tests.unit import base_classes
def test_serialize_and_deserialize_receipt_model(self):
    serialized = self.receipt_handler.serialize(self.exp_receipt)
    receipt = self.receipt_handler.deserialize(serialized)
    self.assertEqual(self.exp_receipt.user_id, receipt.user_id)
    self.assertEqual(self.exp_receipt.id, receipt.id)
    self.assertEqual(self.exp_receipt.issued_at, receipt.issued_at)