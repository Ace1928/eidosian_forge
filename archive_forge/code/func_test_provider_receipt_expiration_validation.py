import datetime
import uuid
from oslo_utils import timeutils
import freezegun
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import receipt_model
from keystone import receipt
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def test_provider_receipt_expiration_validation(self):
    receipt = receipt_model.ReceiptModel()
    receipt.issued_at = utils.isotime(CURRENT_DATE)
    receipt.expires_at = utils.isotime(CURRENT_DATE - DELTA)
    receipt.id = uuid.uuid4().hex
    with freezegun.freeze_time(CURRENT_DATE):
        self.assertRaises(exception.ReceiptNotFound, PROVIDERS.receipt_provider_api._is_valid_receipt, receipt)
    receipt = receipt_model.ReceiptModel()
    receipt.issued_at = utils.isotime(CURRENT_DATE)
    receipt.expires_at = utils.isotime(CURRENT_DATE + DELTA)
    receipt.id = uuid.uuid4().hex
    with freezegun.freeze_time(CURRENT_DATE):
        self.assertIsNone(PROVIDERS.receipt_provider_api._is_valid_receipt(receipt))