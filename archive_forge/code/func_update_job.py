import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def update_job(client, app, parsed_args, update_dict):
    if is_api_v2(app):
        data = client.jobs.update(parsed_args.job, **update_dict).job
    else:
        data = client.job_executions.update(parsed_args.job, **update_dict).job_execution
    return data