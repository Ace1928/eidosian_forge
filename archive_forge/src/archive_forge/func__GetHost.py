from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import connection_profiles
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def _GetHost(self, profile):
    if profile.mysql:
        return profile.mysql.host
    elif profile.postgresql:
        return profile.postgresql.host
    elif profile.cloudsql:
        return profile.cloudsql.publicIp if profile.cloudsql.publicIp else profile.cloudsql.privateIp
    elif profile.alloydb:
        return profile.alloydb.settings.primaryInstanceSettings.privateIp
    elif profile.oracle:
        return profile.oracle.host
    else:
        return None