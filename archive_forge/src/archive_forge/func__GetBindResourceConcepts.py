from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _GetBindResourceConcepts(verb='to bind to'):
    """Build ConceptParser for (un)bind commands resource args."""
    arg_specs = [CreateDevicePresentationSpec(verb, help_text='The gateway device {}.', name='gateway', required=True), CreateDevicePresentationSpec(verb, help_text='The device {} the gateway.', required=True)]
    fallthroughs = {'--device.registry': ['--gateway.registry'], '--gateway.registry': ['--device.registry']}
    return concept_parsers.ConceptParser(arg_specs, fallthroughs)