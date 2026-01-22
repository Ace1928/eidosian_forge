from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetScanRunResourceSpec():
    return concepts.ResourceSpec('websecurityscanner.projects.scanConfigs.scanRuns', resource_name='scan_run', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, scanConfigsId=ScanAttributeConfig(), scanRunsId=ScanRunAttributeConfig())