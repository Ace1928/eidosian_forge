from unittest import mock
import betamax
from betamax import exceptions
import testtools
from keystoneauth1.fixture import keystoneauth_betamax
from keystoneauth1.fixture import serializer
from keystoneauth1.fixture import v2 as v2Fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
Test the fixture's logic, not its monkey-patching.

    The setUp method of our BetamaxFixture monkey-patches the function to
    construct a session. We don't need to test that particular bit of logic
    here so we do not need to call useFixture in our setUp method.
    