from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from xml.etree import ElementTree
import ipaddr
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class BlacklistEntry(object):
    """Instances contain information about individual blacklist entries."""

    def ToYaml(self):
        statements = ['- subnet: %s' % self.subnet]
        if self.description:
            statements.append('  description: %s' % self._SanitizeForYaml(self.description))
        return statements

    def _SanitizeForYaml(self, dirty_str):
        return "'%s'" % dirty_str.replace('\n', ' ')