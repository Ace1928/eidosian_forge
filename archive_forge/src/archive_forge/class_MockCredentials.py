import datetime
import json
import httplib2
from google.auth import aws
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from gslib.tests import testcase
from gslib.utils.wrapped_credentials import WrappedCredentials
import oauth2client
from six import add_move, MovedModule
from six.moves import mock
class MockCredentials(external_account.Credentials):

    def __init__(self, token=None, expiry=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audience = None
        self.expiry = expiry
        self.token = None

        def side_effect(*args, **kwargs):
            del args, kwargs
            self.token = token
        self.refresh = mock.Mock(side_effect=side_effect)

    def retrieve_subject_token():
        pass