import itertools
import pkgutil
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from oslo_utils import importutils
import stevedore
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient import extension as ext
from novaclient.i18n import _
from novaclient import utils
def _make_discovery_manager():
    return stevedore.ExtensionManager('novaclient.extension')