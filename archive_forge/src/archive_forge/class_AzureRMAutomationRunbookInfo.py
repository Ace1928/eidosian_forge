from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAutomationRunbookInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), automation_account_name=dict(type='str', required=True), show_content=dict(type='bool'), tags=dict(type='list', elements='str'))
        self.results = dict()
        self.resource_group = None
        self.name = None
        self.tags = None
        self.automation_account_name = None
        self.show_content = None
        super(AzureRMAutomationRunbookInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec) + ['tags']:
            setattr(self, key, kwargs[key])
        if self.name and self.show_content:
            runbooks = [self.get_content()]
        elif self.name:
            runbooks = [self.get()]
        else:
            runbooks = self.list_by_automaiton_account()
        self.results['automation_runbook'] = [self.to_dict(x) for x in runbooks if x and self.has_tags(x.tags, self.tags)]
        return self.results

    def get_content(self):
        try:
            return self.automation_client.runbook.get(self.resource_group, self.automation_account_name, self.name)
        except ResourceNotFoundError as exc:
            pass

    def get(self):
        try:
            return self.automation_client.runbook.get(self.resource_group, self.automation_account_name, self.name)
        except ResourceNotFoundError as exc:
            pass

    def list_by_automaiton_account(self):
        result = []
        try:
            resp = self.automation_client.runbook.list_by_automation_account(self.resource_group, self.automation_account_name)
            while True:
                result.append(resp.next())
        except StopIteration:
            pass
        except Exception as exc:
            pass
        return result

    def to_dict(self, runbook):
        if not runbook:
            return None
        runbook_dict = dict(id=runbook.id, type=runbook.type, name=runbook.name, tags=runbook.tags, location=runbook.location, runbook_type=runbook.runbook_type, runbook_content_link=runbook.publish_content_link, state=runbook.state, log_verbose=runbook.log_verbose, log_progress=runbook.log_progress, log_activity_trace=runbook.log_activity_trace, job_count=runbook.job_count, output_types=runbook.output_types, last_modified_by=runbook.last_modified_by, last_modified_time=runbook.last_modified_time, creation_time=runbook.creation_time, description=runbook.description)
        return runbook_dict