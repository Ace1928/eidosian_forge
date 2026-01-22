from keystoneauth1.exceptions import http as ka_exceptions
from mistralclient.api import base as mistral_base
from mistralclient.api import client as mistral_client
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class WorkflowConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    resource_getter_name = 'get_workflow_by_identifier'
    expected_exceptions = (exception.EntityNotFound,)