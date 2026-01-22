from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddKubeconfig(parser):
    help_txt = 'Path to the kubeconfig file.\n\nIf not provided, the default at ~/.kube/config will be used.\n'
    parser.add_argument('--kubeconfig', help=help_txt)