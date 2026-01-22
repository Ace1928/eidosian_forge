from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import resources
def ParseTransformerConfigFile(ref, args, req):
    """Reads the json file of with the transformer configs and parse the content to the request message."""
    del ref
    messages = apis.GetMessagesModule('mediaasset', 'v1alpha')
    message_class = messages.Transformer
    if args.create_transformer_configs_file:
        transformer_configs = json.loads(args.create_transformer_configs_file)
        transformer = encoding.DictToMessage(transformer_configs, message_class)
        utils.ValidateMediaAssetMessage(transformer)
        req.transformer = transformer
    if args.IsKnownAndSpecified('labels'):
        req.transformer.labels = encoding.DictToMessage(args.labels, messages.Transformer.LabelsValue)
    return req