from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PublishSummaryToCscc(_messages.Message):
    """Publish the result summary of a DlpJob to [Security Command
  Center](https://cloud.google.com/security-command-center). This action is
  available for only projects that belong to an organization. This action
  publishes the count of finding instances and their infoTypes. The summary of
  findings are persisted in Security Command Center and are governed by
  [service-specific policies for Security Command
  Center](https://cloud.google.com/terms/service-terms). Only a single
  instance of this action can be specified. Compatible with: Inspect
  """