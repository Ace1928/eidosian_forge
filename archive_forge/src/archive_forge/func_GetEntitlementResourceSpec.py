from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetEntitlementResourceSpec():
    return concepts.ResourceSpec('cloudcommerceconsumerprocurement.projects.entitlements', resource_name='entitlement', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, entitlementsId=EntitlementAttributeConfig())