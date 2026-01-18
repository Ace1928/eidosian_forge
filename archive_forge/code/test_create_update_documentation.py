import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
Update manages new conditions added.

        When a new resource is added during updates, the stacks handles the new
        conditions correctly, and doesn't fail to load them while the update is
        still in progress.
        