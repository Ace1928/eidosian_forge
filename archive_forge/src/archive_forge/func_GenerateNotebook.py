from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateNotebook(args):
    """Creates Notebook field for Content Message Create/Update Requests."""
    module = dataplex_api.GetMessageModule()
    kernel_type_field = module.GoogleCloudDataplexV1ContentNotebook
    notebook = module.GoogleCloudDataplexV1ContentNotebook()
    if args.kernel_type:
        notebook.kernelType = kernel_type_field.KernelTypeValueValuesEnum(args.kernel_type)
    return notebook