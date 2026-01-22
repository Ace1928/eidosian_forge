import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
def _get_value_from_the_table(self, table, key):
    """Parses table to get desired value.

        EXAMPLE of the table:
        # +-------------+----------------------------------+
        # |   Property  |              Value               |
        # +-------------+----------------------------------+
        # | description |                                  |
        # |   enabled   |               True               |
        # |      id     | 582df899eabc47018c96713c2f7196ba |
        # |     name    |              admin               |
        # +-------------+----------------------------------+
        """
    lines = table.split('\n')
    for line in lines:
        if '|' in line:
            l_property, l_value = line.split('|')[1:3]
            if l_property.strip() == key:
                return l_value.strip()
    raise ValueError("Property '%s' is missing from the table:\n%s" % (key, table))