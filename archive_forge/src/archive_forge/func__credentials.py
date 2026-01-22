import os
import urllib.parse
import fixtures
import openstack.config
import testtools
def _credentials(self, cloud='devstack-admin'):
    """Retrieves credentials to run functional tests

        Credentials are either read via os-client-config from the environment
        or from a config file ('clouds.yaml'). Environment variables override
        those from the config file.

        devstack produces a clouds.yaml with two named clouds - one named
        'devstack' which has user privs and one named 'devstack-admin' which
        has admin privs. This function will default to getting the
        devstack-admin cloud as that is the current expected behavior.
        """
    os_cfg = openstack.config.OpenStackConfig()
    try:
        found = os_cfg.get_one_cloud(cloud=cloud)
    except Exception:
        found = os_cfg.get_one_cloud()
    return found