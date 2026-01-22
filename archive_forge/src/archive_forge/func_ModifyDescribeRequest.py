from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
def ModifyDescribeRequest(unused_operation_ref, args, req):
    """Check input and construct describe request if needed."""
    operation_name = args.operation
    project_id = properties.VALUES.core.project.Get()
    if operation_name.startswith('operations/projects'):
        return req
    operation_name_with_prefix = 'operations/projects/' + project_id + '/' + operation_name
    req.name = operation_name_with_prefix
    return req