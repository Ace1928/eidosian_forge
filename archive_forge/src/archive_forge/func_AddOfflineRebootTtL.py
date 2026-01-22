from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddOfflineRebootTtL(parser):
    parser.add_argument('--offline-reboot-ttl', type=arg_parsers.Duration(), help='\n      Limits how long a machine can reboot offline(without connection to google)\n      , specified as a duration relative to the machine\'s most-recent connection\n      to google. The parameter should be a ISO 8601 duration string, for example\n      , "1dT1h2m3s".\n      ', hidden=True)