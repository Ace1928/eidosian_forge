from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def SetServiceRequestUpdateHiveMetastoreConfigs(unused_job_ref, args, update_service_req):
    """Modify the Service update request to update, remove, or clear Hive metastore configurations.

  Args:
    unused_ref: A resource ref to the parsed Service resource.
    args: The parsed args namespace from CLI.
    update_service_req: Created Update request for the API call.

  Returns:
    Modified request for the API call.
  """
    hive_metastore_configs = {}
    if args.update_hive_metastore_configs:
        hive_metastore_configs = args.update_hive_metastore_configs
    if args.update_hive_metastore_configs_from_file:
        hive_metastore_configs = LoadHiveMetatsoreConfigsFromXmlFile(args.update_hive_metastore_configs_from_file)
    update_service_req.service.hiveMetastoreConfig.configOverrides = _GenerateAdditionalProperties(hive_metastore_configs)
    return update_service_req