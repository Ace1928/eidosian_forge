from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class EventTriggerFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('name', 'Name'), ('workflow_id', 'Workflow ID'), ('workflow_params', 'Params'), ('exchange', 'Exchange'), ('topic', 'Topic'), ('event', 'Event'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]

    @staticmethod
    def format(trigger=None, lister=False):
        if trigger:
            data = (trigger.id, trigger.name, trigger.workflow_id, trigger.workflow_params, trigger.exchange, trigger.topic, trigger.event, trigger.created_at)
            if hasattr(trigger, 'updated_at'):
                data += (trigger.updated_at,)
            else:
                data += (None,)
        else:
            data = (tuple(('' for _ in range(len(EventTriggerFormatter.COLUMNS)))),)
        return (EventTriggerFormatter.headings(), data)