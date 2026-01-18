from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
Fail to create resource w/o admin role.

        Integration tests job runs as normal OpenStack user,
        and the resources above are configured to require
        admin role in default policy file of Heat.
        