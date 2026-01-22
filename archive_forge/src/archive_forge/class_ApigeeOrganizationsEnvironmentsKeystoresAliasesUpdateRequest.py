from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeystoresAliasesUpdateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeystoresAliasesUpdateRequest object.

  Fields:
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
    ignoreExpiryValidation: Required. Flag that specifies whether to ignore
      expiry validation. If set to `true`, no expiry validation will be
      performed.
    ignoreNewlineValidation: Flag that specifies whether to ignore newline
      validation. If set to `true`, no error is thrown when the file contains
      a certificate chain with no newline between each certificate. Defaults
      to `false`.
    name: Required. Name of the alias. Use the following format in your
      request: `organizations/{org}/environments/{env}/keystores/{keystore}/al
      iases/{alias}`
  """
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 1)
    ignoreExpiryValidation = _messages.BooleanField(2)
    ignoreNewlineValidation = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)