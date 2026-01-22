import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _encoding_for_form(inform):
    if inform == PKI_ASN1_FORM:
        encoding = 'UTF-8'
    elif inform == PKIZ_CMS_FORM:
        encoding = 'hex'
    else:
        raise ValueError(_('"inform" must be one of: %s') % ','.join((PKI_ASN1_FORM, PKIZ_CMS_FORM)))
    return encoding