from openstack import resource
class StackTemplate(resource.Resource):
    base_path = '/stacks/%(stack_name)s/%(stack_id)s/template'
    allow_create = False
    allow_list = False
    allow_fetch = True
    allow_delete = False
    allow_commit = False
    name = resource.URI('stack_name')
    stack_name = resource.URI('_stack_name', alias='name')
    stack_id = resource.URI('stack_id', alternate_id=True)
    description = resource.Body('Description')
    heat_template_version = resource.Body('heat_template_version')
    outputs = resource.Body('outputs', type=dict)
    parameters = resource.Body('parameters', type=dict)
    resources = resource.Body('resources', type=dict)
    parameter_groups = resource.Body('parameter_groups', type=list)
    conditions = resource.Body('conditions', type=dict)