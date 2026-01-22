from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.datastore import datastore_index_xml
from googlecloudsdk.third_party.appengine.tools import cron_xml_parser
from googlecloudsdk.third_party.appengine.tools import dispatch_xml_parser
from googlecloudsdk.third_party.appengine.tools import queue_xml_parser
def _MigrateQueueXML(src, dst):
    """Migration script for queue.xml."""
    xml_str = files.ReadFileContents(src)
    yaml_contents = queue_xml_parser.GetQueueYaml(None, xml_str)
    new_files = {src: None, dst: yaml_contents}
    return MigrationResult(new_files)