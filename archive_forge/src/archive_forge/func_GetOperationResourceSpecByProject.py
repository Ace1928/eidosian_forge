from googlecloudsdk.calliope.concepts import concepts
def GetOperationResourceSpecByProject():
    return concepts.ResourceSpec('auditmanager.projects.locations.operationDetails', resource_name='operation', operationDetailsId=OperationAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)