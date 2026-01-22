from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddRemoteBootstrapFlag(parser):
    parser.add_argument('--remote-bootstrap', hidden=True, default=None, help='Use this flag to pass login parameters to a gcloud instance which will help this gcloud to login. This flag is reserved for bootstrapping remote workstation without access to web browsers, which should be initiated by using the --no-browser. Users should not use this flag directly.')