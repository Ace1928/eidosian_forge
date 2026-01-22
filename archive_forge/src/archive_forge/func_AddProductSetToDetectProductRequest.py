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
def AddProductSetToDetectProductRequest(ref, args, request):
    """Adds productSet field to the detect product request."""
    del ref
    try:
        single_request = request.requests[0]
    except IndexError:
        return request
    product_set_ref = args.CONCEPTS.product_set.Parse()
    product_set_name = product_set_ref.RelativeName()
    single_request = _InstantiateProductSearchParams(single_request)
    single_request.imageContext.productSearchParams.productSet = product_set_name
    return request