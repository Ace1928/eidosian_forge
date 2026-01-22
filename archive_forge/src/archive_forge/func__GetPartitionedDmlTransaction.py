from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import http_wrapper
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
def _GetPartitionedDmlTransaction(session_ref):
    """Creates a transaction for Partitioned DML.

  Args:
    session_ref: Reference to the session.

  Returns:
    TransactionSelector with the id property.
  """
    client = apis.GetClientInstance('spanner', 'v1')
    msgs = apis.GetMessagesModule('spanner', 'v1')
    transaction_options = msgs.TransactionOptions(partitionedDml=msgs.PartitionedDml())
    begin_transaction_req = msgs.BeginTransactionRequest(options=transaction_options)
    req = msgs.SpannerProjectsInstancesDatabasesSessionsBeginTransactionRequest(beginTransactionRequest=begin_transaction_req, session=session_ref.RelativeName())
    resp = client.projects_instances_databases_sessions.BeginTransaction(req)
    return msgs.TransactionSelector(id=resp.id)