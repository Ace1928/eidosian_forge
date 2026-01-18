from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_resets_job_template_values(run_module, admin_user, project, inventory):
    module_args = {'name': 'foo', 'playbook': 'helloworld.yml', 'project': project.name, 'inventory': inventory.name, 'extra_vars': {'foo': 'bar'}, 'job_type': 'run', 'state': 'present', 'forks': 20, 'timeout': 50, 'allow_simultaneous': True, 'ask_limit_on_launch': True, 'ask_execution_environment_on_launch': True, 'ask_forks_on_launch': True, 'ask_instance_groups_on_launch': True, 'ask_job_slice_count_on_launch': True, 'ask_labels_on_launch': True, 'ask_timeout_on_launch': True}
    result = run_module('job_template', module_args, admin_user)
    jt = JobTemplate.objects.get(name='foo')
    assert jt.forks == 20
    assert jt.timeout == 50
    assert jt.allow_simultaneous
    assert jt.ask_limit_on_launch
    assert jt.ask_execution_environment_on_launch
    assert jt.ask_forks_on_launch
    assert jt.ask_instance_groups_on_launch
    assert jt.ask_job_slice_count_on_launch
    assert jt.ask_labels_on_launch
    assert jt.ask_timeout_on_launch
    module_args = {'name': 'foo', 'playbook': 'helloworld.yml', 'project': project.name, 'inventory': inventory.name, 'extra_vars': {'foo': 'bar'}, 'job_type': 'run', 'state': 'present', 'forks': 0, 'timeout': 0, 'allow_simultaneous': False, 'ask_limit_on_launch': False, 'ask_execution_environment_on_launch': False, 'ask_forks_on_launch': False, 'ask_instance_groups_on_launch': False, 'ask_job_slice_count_on_launch': False, 'ask_labels_on_launch': False, 'ask_timeout_on_launch': False}
    result = run_module('job_template', module_args, admin_user)
    assert result['changed']
    jt = JobTemplate.objects.get(name='foo')
    assert jt.forks == 0
    assert jt.timeout == 0
    assert not jt.allow_simultaneous
    assert not jt.ask_limit_on_launch
    assert not jt.ask_execution_environment_on_launch
    assert not jt.ask_forks_on_launch
    assert not jt.ask_instance_groups_on_launch
    assert not jt.ask_job_slice_count_on_launch
    assert not jt.ask_labels_on_launch
    assert not jt.ask_timeout_on_launch