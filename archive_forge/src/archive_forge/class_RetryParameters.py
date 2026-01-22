from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class RetryParameters(object):

    def GetYamlStatementsList(self):
        statements = ['  retry_parameters:']
        field_names = (tag.replace('-', '_') for tag in RETRY_PARAMETER_TAGS)
        for field in field_names:
            field_value = getattr(self, field, None)
            if field_value:
                statements.append('    %s: %s' % (field, field_value))
        return statements