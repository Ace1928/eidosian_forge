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
def _extract_role_assignments_from_response_body(self, r):
    assignments = []
    for assignment in r.json['role_assignments']:
        a = {}
        if 'project' in assignment['scope']:
            a['project_id'] = assignment['scope']['project']['id']
        elif 'domain' in assignment['scope']:
            a['domain_id'] = assignment['scope']['domain']['id']
        elif 'system' in assignment['scope']:
            a['system'] = 'all'
        if 'user' in assignment:
            a['user_id'] = assignment['user']['id']
        elif 'group' in assignment:
            a['group_id'] = assignment['group']['id']
        a['role_id'] = assignment['role']['id']
        assignments.append(a)
    return assignments