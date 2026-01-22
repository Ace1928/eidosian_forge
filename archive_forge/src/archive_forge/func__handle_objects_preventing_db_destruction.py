import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _handle_objects_preventing_db_destruction(self, cursor, parameters, verbosity, autoclobber):
    self.log('There are objects in the old test database which prevent its destruction.\nIf they belong to the test user, deleting the user will allow the test database to be recreated.\nOtherwise, you will need to find and remove each of these objects, or use a different tablespace.\n')
    if self._test_user_create():
        if not autoclobber:
            confirm = input("Type 'yes' to delete user %s: " % parameters['user'])
        if autoclobber or confirm == 'yes':
            try:
                if verbosity >= 1:
                    self.log('Destroying old test user...')
                self._destroy_test_user(cursor, parameters, verbosity)
            except Exception as e:
                self.log('Got an error destroying the test user: %s' % e)
                sys.exit(2)
            try:
                if verbosity >= 1:
                    self.log("Destroying old test database for alias '%s'..." % self.connection.alias)
                self._execute_test_db_destruction(cursor, parameters, verbosity)
            except Exception as e:
                self.log('Got an error destroying the test database: %s' % e)
                sys.exit(2)
        else:
            self.log('Tests cancelled -- test database cannot be recreated.')
            sys.exit(1)
    else:
        self.log("Django is configured to use pre-existing test user '%s', and will not attempt to delete it." % parameters['user'])
        self.log('Tests cancelled -- test database cannot be recreated.')
        sys.exit(1)