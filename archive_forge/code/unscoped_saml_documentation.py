from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
Identity v3 unscoped SAML auth action implementations.

The first step of federated auth is to fetch an unscoped token. From there,
the user can list domains and projects they are allowed to access, and request
a scoped token.