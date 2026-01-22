from googlecloudsdk.calliope.concepts import concepts
def FolderAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='folder', help_text='The folder for the {resource}.')