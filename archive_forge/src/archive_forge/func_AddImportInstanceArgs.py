from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddImportInstanceArgs(parser):
    """Register flags Import Instance command."""
    AddInstanceConcept(parser, 'Arguments and flags that specify the Looker instance you want to import.')
    source_group = parser.add_group(mutex=True, required=True, help='Import Destination - The path and storage where the import will be retrieved from.')
    source_group.add_argument('--source-gcs-uri', metavar='SOURCE_GCS_URI', help='The path to the folder in Google Cloud Storage where the import will be retrieved from. The URI is in the form `gs://bucketName/folderName`.')