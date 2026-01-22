import datetime
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import receipt_model
from keystone import notifications
@MEMOIZE_RECEIPTS
def _validate_receipt(self, receipt_id):
    user_id, methods, issued_at, expires_at = self.driver.validate_receipt(receipt_id)
    receipt = receipt_model.ReceiptModel()
    receipt.user_id = user_id
    receipt.methods = methods
    receipt.expires_at = expires_at
    receipt.mint(receipt_id, issued_at)
    return receipt