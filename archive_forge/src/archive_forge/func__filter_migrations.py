from oslo_utils import uuidutils
from novaclient.tests.functional import base
def _filter_migrations(self, version, migration_type, source_compute):
    """
        Filters migrations by --migration-type and --source-compute.

        :param version: The --os-compute-api-version to use.
        :param migration_type: The type of migrations to filter.
        :param source_compute: The source compute service hostname to filter.
        :return: output of the nova migration-list command with filters applied
        """
    return self.nova('migration-list', flags='--os-compute-api-version %s' % version, params='--migration-type %s --source-compute %s' % (migration_type, source_compute))