from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddDestinationAddressFlag(parser):
    """Adds a --destination-addresss flag to the given parser."""
    help_text = 'Destination address for connection.'
    parser.add_argument('--destination-address', help=help_text, required=True)