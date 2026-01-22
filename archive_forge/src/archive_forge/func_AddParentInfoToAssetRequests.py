from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import resources
def AddParentInfoToAssetRequests(ref, args, req):
    """Python hook for yaml commands to wildcard the parent parameter in asset requests."""
    del ref
    project = utils.GetProject()
    location = utils.GetLocation(args)
    req.parent = utils.GetAssetTypeParentTemplate(project, location, args.asset_type)
    return req