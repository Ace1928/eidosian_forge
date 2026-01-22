from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def GenerateAuxiliaryVersionsConfigFromList(unused_ref, args, req):
    """Generates the auxiliary versions map from the list of auxiliary versions.

  Args:
    args: The request arguments.
    req: A request with `service` field.

  Returns:
    If `auxiliary-versions` is present in the arguments, a request with hive
    metastore config's auxiliary versions map field is returned.
    Otherwise the original request is returned.
  """
    if args.auxiliary_versions:
        if req.service.hiveMetastoreConfig is None:
            req.service.hiveMetastoreConfig = {}
        req.service.hiveMetastoreConfig.auxiliaryVersions = _GenerateAuxiliaryVersionsVersionList(args.auxiliary_versions)
    return req