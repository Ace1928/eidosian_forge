from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeystoresAliasesGetCertificateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeystoresAliasesGetCertificateRequest
  object.

  Fields:
    name: Required. Name of the alias. Use the following format in your
      request: `organizations/{org}/environments/{env}/keystores/{keystore}/al
      iases/{alias}`.
  """
    name = _messages.StringField(1, required=True)