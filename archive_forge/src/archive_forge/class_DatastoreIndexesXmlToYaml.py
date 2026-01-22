from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import migrate_config
class DatastoreIndexesXmlToYaml(base.Command):
    """Convert a datastore-indexes.xml file to index.yaml."""

    @staticmethod
    def Args(parser):
        parser.add_argument('xml_file', help='Path to the datastore-indexes.xml file.')
        parser.add_argument('--generated-indexes-file', help='If specified, include the auto-generated xml file too, and merge the resulting entries appropriately. Note that this file is usually named `WEB-INF/appengine-generated/datastore-indexes-auto.xml`.')

    def Run(self, args):
        src = os.path.abspath(args.xml_file)
        dst = os.path.join(os.path.dirname(src), 'index.yaml')
        auto_src = None
        if args.generated_indexes_file:
            auto_src = os.path.abspath(args.generated_indexes_file)
        entry = migrate_config.REGISTRY['datastore-indexes-xml-to-yaml']
        migrate_config.Run(entry, src=src, dst=dst, auto_src=auto_src)