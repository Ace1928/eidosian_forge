from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetCurrentServerCa(sql_client, sql_messages, instance_ref):
    """Returns the currently active Server CA Cert."""
    server_ca_types = GetServerCaTypeDict(ListServerCas(sql_client, sql_messages, instance_ref))
    return server_ca_types.get(ACTIVE_CERT_LABEL)