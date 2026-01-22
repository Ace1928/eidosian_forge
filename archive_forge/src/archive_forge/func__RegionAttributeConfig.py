from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def _RegionAttributeConfig():
    fallthroughs = [deps.PropertyFallthrough(properties.VALUES.dataproc.region)]
    return concepts.ResourceParameterAttributeConfig(name='region', help_text='Dataproc region for the {resource}. Each Dataproc region constitutes an independent resource namespace constrained to deploying instances into Compute Engine zones inside the region. Overrides the default `dataproc/region` property value for this command invocation.', fallthroughs=fallthroughs)