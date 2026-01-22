from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from mistralclient.api import base
class CronTriggerManager(base.ResourceManager):
    resource_class = CronTrigger

    def create(self, name, workflow_identifier, workflow_input=None, workflow_params=None, pattern=None, first_time=None, count=None):
        self._ensure_not_empty(name=name, workflow_identifier=workflow_identifier)
        data = {'name': name, 'pattern': pattern, 'first_execution_time': first_time, 'remaining_executions': count}
        if uuidutils.is_uuid_like(workflow_identifier):
            data.update({'workflow_id': workflow_identifier})
        else:
            data.update({'workflow_name': workflow_identifier})
        if workflow_input:
            data.update({'workflow_input': jsonutils.dumps(workflow_input)})
        if workflow_params:
            data.update({'workflow_params': jsonutils.dumps(workflow_params)})
        return self._create('/cron_triggers', data)

    def list(self, marker='', limit=None, sort_keys='', fields='', sort_dirs='', **filters):
        query_string = self._build_query_params(marker=marker, limit=limit, sort_keys=sort_keys, sort_dirs=sort_dirs, fields=fields, filters=filters)
        return self._list('/cron_triggers%s' % query_string, response_key='cron_triggers')

    def get(self, name):
        self._ensure_not_empty(name=name)
        return self._get('/cron_triggers/%s' % name)

    def delete(self, name):
        self._ensure_not_empty(name=name)
        self._delete('/cron_triggers/%s' % name)