from oslo_log import log as logging
from saharaclient.osc.v1 import job_templates as jt_v1
class ListJobTemplates(jt_v1.ListJobTemplates):
    """Lists job templates"""
    log = logging.getLogger(__name__ + '.ListJobTemplates')