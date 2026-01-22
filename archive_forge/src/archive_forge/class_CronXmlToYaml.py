from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import migrate_config
class CronXmlToYaml(base.Command):
    """Convert a cron.xml file to cron.yaml."""

    @staticmethod
    def Args(parser):
        parser.add_argument('xml_file', help='Path to the cron.xml file.')

    def Run(self, args):
        src = os.path.abspath(args.xml_file)
        dst = os.path.join(os.path.dirname(src), 'cron.yaml')
        entry = migrate_config.REGISTRY['cron-xml-to-yaml']
        migrate_config.Run(entry, src=src, dst=dst)