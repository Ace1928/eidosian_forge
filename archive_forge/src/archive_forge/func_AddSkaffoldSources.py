from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSkaffoldSources(parser):
    """Add Skaffold sources."""
    skaffold_source_config_group = parser.add_mutually_exclusive_group()
    skaffold_source_group = skaffold_source_config_group.add_group(mutex=False)
    AddSkaffoldFileFlag().AddToParser(skaffold_source_group)
    AddSourceFlag().AddToParser(skaffold_source_group)
    AddKubernetesFileFlag().AddToParser(skaffold_source_config_group)
    AddCloudRunFileFlag().AddToParser(skaffold_source_config_group)
    run_container_group = skaffold_source_config_group.add_group(mutex=False, hidden=True)
    AddFromRunContainerFlag().AddToParser(run_container_group)
    AddServicesFlag().AddToParser(run_container_group)