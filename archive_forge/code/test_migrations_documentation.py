import collections
import os
from alembic import command as alembic_command
from alembic import script as alembic_script
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from sqlalchemy import sql
import sqlalchemy.types as types
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import versions
from glance.db.sqlalchemy import models
from glance.db.sqlalchemy import models_metadef
import glance.tests.utils as test_utils
Test that migrations follow the conventional rules.

        Each release should have at least one file for each of the required
        phases, if it has one for any of them. They should also be named
        in a consistent way going forward.
        