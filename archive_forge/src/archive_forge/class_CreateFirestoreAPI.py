from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.firestore import flags
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class CreateFirestoreAPI(base.Command):
    """Create a Google Cloud Firestore database via Firestore API.

  ## EXAMPLES

  To create a Firestore Native database in `nam5`.

      $ {command} --location=nam5

  To create a Datastore Mode database in `us-east1`.

      $ {command} --location=us-east1 --type=datastore-mode

  To create a Datastore Mode database in `us-east1` with a databaseId `foo`.

      $ {command} --database=foo --location=us-east1 --type=datastore-mode

  To create a Firestore Native database in `nam5` with delete protection
  enabled.

      $ {command} --location=nam5 --delete-protection

  To create a Firestore Native database in `nam5` with Point In Time Recovery
  (PITR) enabled.

      $ {command} --location=nam5 --enable-pitr
  """

    def DatabaseType(self, database_type):
        if database_type == 'firestore-native':
            return api_utils.GetMessages().GoogleFirestoreAdminV1Database.TypeValueValuesEnum.FIRESTORE_NATIVE
        elif database_type == 'datastore-mode':
            return api_utils.GetMessages().GoogleFirestoreAdminV1Database.TypeValueValuesEnum.DATASTORE_MODE
        else:
            raise ValueError('invalid database type: {}'.format(database_type))

    def DatabaseDeleteProtectionState(self, enable_delete_protection):
        if enable_delete_protection:
            return api_utils.GetMessages().GoogleFirestoreAdminV1Database.DeleteProtectionStateValueValuesEnum.DELETE_PROTECTION_ENABLED
        return api_utils.GetMessages().GoogleFirestoreAdminV1Database.DeleteProtectionStateValueValuesEnum.DELETE_PROTECTION_DISABLED

    def DatabasePitrState(self, enable_pitr):
        if enable_pitr is None:
            return api_utils.GetMessages().GoogleFirestoreAdminV1Database.PointInTimeRecoveryEnablementValueValuesEnum.POINT_IN_TIME_RECOVERY_ENABLEMENT_UNSPECIFIED
        if enable_pitr:
            return api_utils.GetMessages().GoogleFirestoreAdminV1Database.PointInTimeRecoveryEnablementValueValuesEnum.POINT_IN_TIME_RECOVERY_ENABLED
        return api_utils.GetMessages().GoogleFirestoreAdminV1Database.PointInTimeRecoveryEnablementValueValuesEnum.POINT_IN_TIME_RECOVERY_DISABLED

    def DatabaseCmekConfig(self, args):
        return api_utils.GetMessages().GoogleFirestoreAdminV1CmekConfig()

    def Run(self, args):
        project = properties.VALUES.core.project.Get(required=True)
        return databases.CreateDatabase(project, args.location, args.database, self.DatabaseType(args.type), self.DatabaseDeleteProtectionState(args.delete_protection), self.DatabasePitrState(args.enable_pitr), self.DatabaseCmekConfig(args))

    @classmethod
    def Args(cls, parser):
        flags.AddLocationFlag(parser, required=True, suggestion_aliases=['--region'])
        parser.add_argument('--type', help='The type of the database.', default='firestore-native', choices=['firestore-native', 'datastore-mode'])
        parser.add_argument('--database', help='The ID to use for the database, which will become the final\n        component of the database\'s resource name. If database ID is not\n        provided, (default) will be used as database ID.\n\n        This value should be 4-63 characters. Valid characters are /[a-z][0-9]-/\n        with first character a letter and the last a letter or a number. Must\n        not be UUID-like /[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}/.\n\n        Using "(default)" database ID is also allowed.\n        ', type=str, default='(default)')
        parser.add_argument('--delete-protection', help='Whether to enable delete protection on the created database.\n\n        If set to true, delete protection of the new database will be enabled\n        and delete operations will fail unless delete protection is disabled.\n\n        Default to false.\n        ', action='store_true', default=False)
        parser.add_argument('--enable-pitr', help='Whether to enable Point In Time Recovery (PITR) on the created\n        database.\n\n        If set to true, PITR on the new database will be enabled. By default,\n        this feature is not enabled.\n        ', action='store_true', default=None)