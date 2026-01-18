from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplate, NotificationTemplate
@pytest.mark.django_db
def test_survey_spec_missing_field(run_module, admin_user, organization, survey_spec):
    wfjt = WorkflowJobTemplate.objects.create(organization=organization, name='foo-workflow', survey_enabled=True, survey_spec=survey_spec)
    result = run_module('workflow_job_template', {'name': 'foo-workflow', 'organization': organization.name, 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert not result.get('changed', True), result
    wfjt.refresh_from_db()
    assert wfjt.survey_spec == survey_spec
    del survey_spec['description']
    result = run_module('workflow_job_template', {'name': 'foo-workflow', 'organization': organization.name, 'survey_spec': survey_spec, 'state': 'present'}, admin_user)
    assert result.get('failed', True)
    assert result.get('msg') == "Failed to update survey: Field 'description' is missing from survey spec."