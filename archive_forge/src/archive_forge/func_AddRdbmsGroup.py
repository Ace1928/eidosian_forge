from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
def AddRdbmsGroup(parser):
    """Adds necessary RDBMS params for discover command parser."""
    rdbms_parser = parser.add_group(mutex=True)
    rdbms_parser.add_argument('--mysql-rdbms-file', help='Path to a YAML (or JSON) file containing the MySQL RDBMS to enrich with child data objects and metadata. If you pass - as the value of the flag the file content will be read from stdin. ')
    rdbms_parser.add_argument('--oracle-rdbms-file', help='Path to a YAML (or JSON) file containing the Oracle RDBMS to enrich with child data objects and metadata. If you pass - as the value of the flag the file content will be read from stdin.')
    rdbms_parser.add_argument('--postgresql-rdbms-file', help='Path to a YAML (or JSON) file containing the PostgreSQL RDBMS to enrich with child data objects and metadata. If you pass - as the value of the flag the file content will be read from stdin.')