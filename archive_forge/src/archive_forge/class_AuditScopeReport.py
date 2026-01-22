from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditScopeReport(_messages.Message):
    """Response message containing the Audit Scope Report.

  Fields:
    name: Identifier. The name of this Audit Report, in the format of scope
      given in request.
    scopeReportContents: Audit Scope report content in byte format.
  """
    name = _messages.StringField(1)
    scopeReportContents = _messages.BytesField(2)