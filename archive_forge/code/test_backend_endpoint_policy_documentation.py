import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
Create sample data to test policy associations.

        The following data is created:

        - 3 regions, in a hierarchy, 0 -> 1 -> 2 (where 0 is top)
        - 3 services
        - 6 endpoints, 2 in each region, with a mixture of services:
          0 - region 0, Service 0
          1 - region 0, Service 1
          2 - region 1, Service 1
          3 - region 1, Service 2
          4 - region 2, Service 2
          5 - region 2, Service 0

        