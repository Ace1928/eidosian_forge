import copy
import http.client
import uuid
from oslo_serialization import jsonutils
from keystone.common.policies import role_assignment as rp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
Common functionality for domain users.