from unittest import mock
from heat.common import exception
from heat.db import api as db_api
from heat.tests import utils
def all_resources(self):
    try:
        resources = db_api.resource_get_all(self.cntxt)
    except exception.NotFound:
        return []
    ret = []
    for res in resources:
        if res.action in ('CREATE', 'UPDATE') and res.status == 'COMPLETE':
            ret.append(res)
    return ret