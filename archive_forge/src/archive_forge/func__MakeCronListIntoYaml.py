from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
def _MakeCronListIntoYaml(cron_list):
    """Converts list of yaml statements describing cron jobs into a string."""
    statements = ['cron:']
    for cron in cron_list:
        statements += cron.ToYaml()
    return '\n'.join(statements) + '\n'