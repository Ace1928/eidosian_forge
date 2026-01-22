from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformParamsSpec(ps):
    """Convert ParamsSpecs into Tekton yaml."""
    param_spec = []
    for p in ps:
        param = {}
        if 'name' in p:
            param['name'] = p.pop('name')
        if 'description' in p:
            param['description'] = p.pop('description')
        if 'type' in p:
            param['type'] = p.pop('type').lower()
        if 'default' in p:
            param['default'] = _TransformParamValue(p.pop('default'))
        if 'properties' in p:
            param['properties'] = p.pop('properties')
        param_spec.append(param)
    return param_spec