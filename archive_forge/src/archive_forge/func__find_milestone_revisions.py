import copy
import sys
from alembic import command as alembic_command
from alembic import script as alembic_script
from alembic import util as alembic_util
from oslo_config import cfg
from oslo_log import log
import pbr.version
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.i18n import _
def _find_milestone_revisions(config, milestone, branch=None):
    """Return the revision(s) for a given milestone."""
    script = alembic_script.ScriptDirectory.from_config(config)
    return [(m.revision, label) for m in _get_revisions(script) for label in m.branch_labels or [None] if milestone in getattr(m.module, 'keystone_milestone', []) and (branch is None or branch in m.branch_labels)]