from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from xml.etree import ElementTree
import ipaddr
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def _MakeDosListIntoYaml(dos_list):
    """Converts yaml statement list of blacklisted IP's into a string."""
    statements = ['blacklist:']
    for entry in dos_list:
        statements += entry.ToYaml()
    return '\n'.join(statements) + '\n'