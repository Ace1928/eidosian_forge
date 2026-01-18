from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (

        A custom C{lp_save} method can be set on a L{FakeResource} after its
        been created.
        