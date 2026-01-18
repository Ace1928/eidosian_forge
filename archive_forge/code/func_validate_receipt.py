import os
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.receipt.providers import base
from keystone.receipt import receipt_formatters as tf
def validate_receipt(self, receipt_id):
    try:
        return self.receipt_formatter.validate_receipt(receipt_id)
    except exception.ValidationError:
        raise exception.ReceiptNotFound(receipt_id=receipt_id)