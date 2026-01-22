from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def LoadHiveMetatsoreConfigsFromXmlFile(file_arg):
    """Convert Input XML file into Hive metastore configurations."""
    hive_metastore_configs = {}
    root = element_tree.fromstring(file_arg)
    for prop in root.iter('property'):
        hive_metastore_configs[prop.find('name').text] = prop.find('value').text
    return hive_metastore_configs