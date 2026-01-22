from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from urllib import parse
def actions_to_urls(self, props):
    kwargs = dict(props)

    def get_urls(action_type, queue_type):
        for act in kwargs.get(action_type) or []:
            if act in self.stack:
                yield self.stack[act].FnGetAtt('AlarmUrl')
            elif act:
                yield act
        for queue in kwargs.pop(queue_type, []):
            query = {'queue_name': queue}
            yield ('trust+zaqar://?%s' % parse.urlencode(query))
    action_props = {arg_types[0]: list(get_urls(*arg_types)) for arg_types in ((ALARM_ACTIONS, ALARM_QUEUES), (OK_ACTIONS, OK_QUEUES), (INSUFFICIENT_DATA_ACTIONS, INSUFFICIENT_DATA_QUEUES))}
    kwargs.update(action_props)
    return kwargs