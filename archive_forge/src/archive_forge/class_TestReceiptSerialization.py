import datetime
from unittest import mock
import uuid
from keystone.common.cache import _context_cache
from keystone.common import utils as ks_utils
from keystone import exception
from keystone.models import receipt_model
from keystone.tests.unit import base_classes
class TestReceiptSerialization(base_classes.TestCaseWithBootstrap):

    def setUp(self):
        super(TestReceiptSerialization, self).setUp()
        self.admin_user_id = self.bootstrapper.admin_user_id
        self.receipt_id = uuid.uuid4().hex
        issued_at = datetime.datetime.utcnow()
        self.issued_at = ks_utils.isotime(at=issued_at, subsecond=True)
        receipt_handler_id = receipt_model._ReceiptModelHandler.identity
        self.receipt_handler = _context_cache._registry.get(receipt_handler_id)
        self.exp_receipt = receipt_model.ReceiptModel()
        self.exp_receipt.user_id = self.admin_user_id
        self.exp_receipt.mint(self.receipt_id, self.issued_at)

    def test_serialize_and_deserialize_receipt_model(self):
        serialized = self.receipt_handler.serialize(self.exp_receipt)
        receipt = self.receipt_handler.deserialize(serialized)
        self.assertEqual(self.exp_receipt.user_id, receipt.user_id)
        self.assertEqual(self.exp_receipt.id, receipt.id)
        self.assertEqual(self.exp_receipt.issued_at, receipt.issued_at)

    @mock.patch.object(receipt_model.ReceiptModel, '__init__', side_effect=Exception)
    def test_error_handling_in_deserialize(self, handler_mock):
        serialized = self.receipt_handler.serialize(self.exp_receipt)
        self.assertRaises(exception.CacheDeserializationError, self.receipt_handler.deserialize, serialized)