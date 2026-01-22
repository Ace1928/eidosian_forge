from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from datetime import datetime
class GitlabMilestones(object):

    def __init__(self, module, gitlab_instance, group_id, project_id):
        self._gitlab = gitlab_instance
        self.gitlab_object = group_id if group_id else project_id
        self.is_group_milestone = True if group_id else False
        self._module = module

    def list_all_milestones(self):
        page_nb = 1
        milestones = []
        vars_page = self.gitlab_object.milestones.list(page=page_nb)
        while len(vars_page) > 0:
            milestones += vars_page
            page_nb += 1
            vars_page = self.gitlab_object.milestones.list(page=page_nb)
        return milestones

    def create_milestone(self, var_obj):
        if self._module.check_mode:
            return (True, True)
        var = {'title': var_obj.get('title')}
        if var_obj.get('description') is not None:
            var['description'] = var_obj.get('description')
        if var_obj.get('start_date') is not None:
            var['start_date'] = self.check_date(var_obj.get('start_date'))
        if var_obj.get('due_date') is not None:
            var['due_date'] = self.check_date(var_obj.get('due_date'))
        _obj = self.gitlab_object.milestones.create(var)
        return (True, _obj.asdict())

    def update_milestone(self, var_obj):
        if self._module.check_mode:
            return (True, True)
        _milestone = self.gitlab_object.milestones.get(self.get_milestone_id(var_obj.get('title')))
        if var_obj.get('description') is not None:
            _milestone.description = var_obj.get('description')
        if var_obj.get('start_date') is not None:
            _milestone.start_date = var_obj.get('start_date')
        if var_obj.get('due_date') is not None:
            _milestone.due_date = var_obj.get('due_date')
        _milestone.save()
        return (True, _milestone.asdict())

    def get_milestone_id(self, _title):
        _milestone_list = self.gitlab_object.milestones.list()
        _found = list(filter(lambda x: x.title == _title, _milestone_list))
        if _found:
            return _found[0].id
        else:
            self._module.fail_json(msg="milestone '%s' not found." % _title)

    def check_date(self, _date):
        try:
            datetime.strptime(_date, '%Y-%m-%d')
        except ValueError:
            self._module.fail_json(msg="milestone's date '%s' not in correct format." % _date)
        return _date

    def delete_milestone(self, var_obj):
        if self._module.check_mode:
            return (True, True)
        _milestone = self.gitlab_object.milestones.get(self.get_milestone_id(var_obj.get('title')))
        _milestone.delete()
        return (True, _milestone.asdict())