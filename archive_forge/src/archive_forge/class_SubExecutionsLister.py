import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient.commands.v2 import executions
from mistralclient import utils
class SubExecutionsLister(executions.SubExecutionsBaseLister):

    def _get_format_function(self):
        return executions.ExecutionFormatter.format_list

    def _get_resources_function(self):
        mistral_client = self.app.client_manager.workflow_engine
        return mistral_client.tasks.get_task_sub_executions