from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddStateFlag(parser):
    """Adds a --state flag to the given parser."""
    help_text = 'Stream state, can be set to: "RUNNING" or "PAUSED".'
    parser.add_argument('--state', help=help_text)