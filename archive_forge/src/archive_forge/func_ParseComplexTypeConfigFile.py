from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def ParseComplexTypeConfigFile(ref, args, req):
    """Reads the json with complex type configuration and set the content in the request."""
    del ref
    complex_type_dict = []
    if args.complex_type_config_file:
        complex_type_dict = json.loads(args.complex_type_config_file)
        messages = utils.GetApiMessage(utils.GetApiVersionFromArgs(args))
        ct = encoding.DictToMessage(complex_type_dict, messages.ComplexType)
        utils.ValidateMediaAssetMessage(ct)
        req.complexType = ct
    if 'update' in args.command_path:
        ValidateUpdateMask(args, complex_type_dict)
    return req