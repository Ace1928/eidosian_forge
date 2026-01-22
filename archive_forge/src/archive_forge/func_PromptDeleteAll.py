from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.ml.vision import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
def PromptDeleteAll(ref, args, request):
    """Prompts to confirm deletion. Changes orphan-products to None if False."""
    del ref
    if not args.force:
        console_io.PromptContinue(message='You are about to delete products. After deletion, the products cannot be restored.', cancel_on_no=True)
        request.purgeProductsRequest.force = True
    if args.product_set:
        request.purgeProductsRequest.deleteOrphanProducts = None
    return request