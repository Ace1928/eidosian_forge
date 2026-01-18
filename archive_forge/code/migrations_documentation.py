from novaclient import api_versions
from novaclient import base

        Get a list of migrations.
        :param host: filter migrations by host name (optional).
        :param status: filter migrations by status (optional).
        :param instance_uuid: filter migrations by instance uuid (optional).
        :param marker: Begin returning migrations that appear later in the
        migrations list than that represented by this migration UUID
        (optional).
        :param limit: maximum number of migrations to return (optional).
        Note the API server has a configurable default limit. If no limit is
        specified here or limit is larger than default, the default limit will
        be used.
        :param changes_since: Only return migrations changed later or equal
        to a certain point of time. The provided time should be an ISO 8061
        formatted time. e.g. 2016-03-04T06:27:59Z . (optional).
        :param changes_before: Only return migrations changed earlier or
        equal to a certain point of time. The provided time should be an ISO
        8061 formatted time. e.g. 2016-03-05T06:27:59Z . (optional).
        :param migration_type: Filter migrations by type. Valid values are:
        evacuation, live-migration, migration, resize
        :param source_compute: Filter migrations by source compute host name.
        :param user_id: filter migrations by user (optional).
        :param project_id: filter migrations by project (optional).
        