from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1TransactionDataUser(_messages.Message):
    """Details about a user's account involved in the transaction.

  Fields:
    accountId: Optional. Unique account identifier for this user. If using
      account defender, this should match the hashed_account_id field.
      Otherwise, a unique and persistent identifier for this account.
    creationMs: Optional. The epoch milliseconds of the user's account
      creation.
    email: Optional. The email address of the user.
    emailVerified: Optional. Whether the email has been verified to be
      accessible by the user (OTP or similar).
    phoneNumber: Optional. The phone number of the user, with country code.
    phoneVerified: Optional. Whether the phone number has been verified to be
      accessible by the user (OTP or similar).
  """
    accountId = _messages.StringField(1)
    creationMs = _messages.IntegerField(2)
    email = _messages.StringField(3)
    emailVerified = _messages.BooleanField(4)
    phoneNumber = _messages.StringField(5)
    phoneVerified = _messages.BooleanField(6)