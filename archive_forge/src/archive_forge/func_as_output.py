import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def as_output(self, resource_name, template_type='cfn'):
    """Output entry for a provider template with the given resource name.

        :param resource_name: the logical name of the provider resource
        :param template_type: the template type to generate
        :returns: This attribute as a template 'Output' entry for
                  cfn template and 'output' entry for hot template
        """
    if template_type == 'hot':
        return {'value': {'get_attr': [resource_name, self.name]}, 'description': self.schema.description}
    else:
        return {'Value': {'Fn::GetAtt': [resource_name, self.name]}, 'Description': self.schema.description}