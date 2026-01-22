from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddSystemAddonsConfig(parser):
    parser.add_argument('--system-addons-config', type=arg_parsers.YAMLFileContents(), help='\n      If specified as a YAML/JSON file, customized configuration in this file\n      will be applied to the system add-ons.\n\n      For example,\n\n      {\n        "systemAddonsConfig": {\n          "ingress": {\n            "disabled": true,\n            "ipv4_vip": "10.0.0.1"\n          }\n        }\n      }\n      ')