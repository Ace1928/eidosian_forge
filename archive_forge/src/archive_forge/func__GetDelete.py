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
@classmethod
def _GetDelete(cls, table, keys):
    """Constructs Delete object, which is needed for delete operation."""

    def _ToJson(msg):
        return extra_types.JsonProtoEncoder(extra_types.JsonArray(entries=msg.entry))
    encoding.RegisterCustomMessageCodec(encoder=_ToJson, decoder=None)(cls.msgs.KeySet.KeysValueListEntry)
    key_set = cls.msgs.KeySet(keys=[cls.msgs.KeySet.KeysValueListEntry(entry=table.GetJsonKeys(keys))])
    return cls.msgs.Delete(table=table.name, keySet=key_set)