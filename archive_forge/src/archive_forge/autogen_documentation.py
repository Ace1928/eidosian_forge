from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
Generate a new ops.MigrationScript() for a given phase.

    E.g. given an ops.MigrationScript() directive from a vanilla autogenerate
    and an expand/contract phase name, produce a new ops.MigrationScript()
    which contains only those sub-directives appropriate to "expand" or
    "contract".  Also ensure that the branch directory exists and that
    the correct branch labels/depends_on/head revision are set up.
    