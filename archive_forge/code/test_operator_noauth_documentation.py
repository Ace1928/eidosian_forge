import openstack.cloud
from openstack.tests.unit import base
Test noauth selection for Ironic in OpenStackCloud

        Sometimes people also write clouds.yaml files that look like this:

        ::
          clouds:
            bifrost:
              auth_type: "none"
              endpoint: https://baremetal.example.com
        