from oslo_config import cfg
from saharaclient.api import base as sahara_base
from saharaclient import client as sahara_client
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
Find the job type

        :param job_type: the name of sahara job type to find
        :returns: the name of :job_type:
        :raises: exception.EntityNotFound
        