from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import stream_objects
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datastream import resource_args
from googlecloudsdk.core import properties
def _GetSourceObject(self, stream_object):
    if stream_object.sourceObject.mysqlIdentifier:
        identifier = stream_object.sourceObject.mysqlIdentifier
        return '%s.%s' % (identifier.database, identifier.table)
    elif stream_object.sourceObject.oracleIdentifier:
        identifier = stream_object.sourceObject.oracleIdentifier
        return '%s.%s' % (identifier.schema, identifier.table)
    elif stream_object.sourceObject.postgresqlIdentifier:
        identifier = stream_object.sourceObject.postgresqlIdentifier
        return '%s.%s' % (identifier.schema, identifier.table)
    else:
        return None