from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def _ProcessPullQueueNode(self, node, queue):
    """Populates PullQueue-specific fields from parsed XML."""
    for tag in PUSH_QUEUE_TAGS:
        if xml_parser_utils.GetChild(node, tag) is not None:
            self.errors.append(PULL_QUEUE_ERROR_MESSAGE % (tag, queue.name))
    acl_node = xml_parser_utils.GetChild(node, 'acl')
    if acl_node is not None:
        queue.acl = Acl()
        queue.acl.user_emails = [sub_node.text for sub_node in xml_parser_utils.GetNodes(acl_node, 'user-email')]
        queue.acl.writer_emails = [sub_node.text for sub_node in xml_parser_utils.GetNodes(acl_node, 'writer-email')]
    else:
        queue.acl = None
    self._ProcessRetryParametersNode(node, queue)