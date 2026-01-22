from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
def AddMysqlProfileGroup(parser, required=True):
    """Adds necessary mysql profile flags to the given parser."""
    mysql_profile = parser.add_group()
    mysql_profile.add_argument('--mysql-hostname', help='IP or hostname of the MySQL source database.', required=required)
    mysql_profile.add_argument('--mysql-port', help='Network port of the MySQL source database.', required=required, type=int)
    mysql_profile.add_argument('--mysql-username', help='Username Datastream will use to connect to the database.', required=required)
    password_group = mysql_profile.add_group(required=required, mutex=True)
    password_group.add_argument('--mysql-password', help='          Password for the user that Datastream will be using to\n          connect to the database.\n          This field is not returned on request, and the value is encrypted\n          when stored in Datastream.')
    password_group.add_argument('--mysql-prompt-for-password', action='store_true', help='Prompt for the password used to connect to the database.')
    ssl_config = mysql_profile.add_group()
    ssl_config.add_argument('--ca-certificate', help="          x509 PEM-encoded certificate of the CA that signed the source database\n          server's certificate. The replica will use this certificate to verify\n          it's connecting to the right host.", required=required)
    ssl_config.add_argument('--client-certificate', help='          x509 PEM-encoded certificate that will be used by the replica to\n          authenticate against the source database server.', required=required)
    ssl_config.add_argument('--client-key', help='          Unencrypted PKCS#1 or PKCS#8 PEM-encoded private key associated with\n          the Client Certificate.', required=required)