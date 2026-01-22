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
class TestVersions(test_utils.BaseTestCase):

    def test_phase_and_naming(self):
        """Test that migrations follow the conventional rules.

        Each release should have at least one file for each of the required
        phases, if it has one for any of them. They should also be named
        in a consistent way going forward.
        """
        required_phases = set(['expand', 'migrate', 'contract'])
        exception_releases = ['liberty', 'mitaka']
        versions_path, _ = os.path.split(versions.__file__)
        version_files = os.listdir(versions_path)
        version_files += os.listdir(os.path.join(versions_path, '..', 'data_migrations'))
        releases = collections.defaultdict(set)
        for version_file in [v for v in version_files if v[0] != '_']:
            if any([version_file.startswith(prefix) for prefix in exception_releases]):
                continue
            if not version_file.split('_', 2)[0].isnumeric():
                try:
                    _rest = ''
                    release, phasever, _rest = version_file.split('_', 2)
                except ValueError:
                    release = phasever = ''
                phase = ''.join((x for x in phasever if x.isalpha()))
                if phase not in required_phases:
                    self.fail('Migration files should be in the form of: release_phaseNN_some_description.py (while processing %r)' % version_file)
                releases[release].add(phase)
            else:
                try:
                    _rest = ''
                    release_y, release_n, phasever, _rest = version_file.split('_', 3)
                except ValueError:
                    release_y = phasever = ''
                phase = ''.join((x for x in phasever if x.isalpha()))
                if phase not in required_phases:
                    self.fail('Migration files should be in the form of: releaseYear_releaseN_phaseNN_description.py (while processing %r)' % version_file)
                releases[release_y].add(phase)
        for release, phases in releases.items():
            missing = required_phases - phases
            if missing:
                self.fail('Release %s missing migration phases %s' % (release, ','.join(missing)))