from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def _ValidateSpecAndAttributeExist(self, location, spec_name, attribute_name):
    """Raises if a formatted string refers to non-existent spec or attribute."""
    if spec_name not in self.specs:
        raise ValueError('invalid fallthrough {}: [{}]. Spec name is not present in the presentation specs. Available names: [{}]'.format(location, '{}.{}'.format(spec_name, attribute_name), ', '.join(sorted(list(self.specs.keys())))))
    spec = self.specs.get(spec_name)
    if attribute_name not in [attribute.name for attribute in spec.concept_spec.attributes]:
        raise ValueError('invalid fallthrough {}: [{}]. spec named [{}] has no attribute named [{}]'.format(location, '{}.{}'.format(spec_name, attribute_name), spec_name, attribute_name))