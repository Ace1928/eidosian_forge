import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
class ScheduleExecution(base.Resource):
    """ScheduleExecution is a resource used to hold information about
    the execution of a scheduled backup.
    """

    def __repr__(self):
        return '<Execution: %s>' % self.name