from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddMetalLbConfig(lb_config_mutex_group):
    """Adds flags for MetalLB load balancer.

  Args:
    lb_config_mutex_group: The parent mutex group to add the flags to.
  """
    metal_lb_config_mutex_group = lb_config_mutex_group.add_group(help='MetalLB Configuration', mutex=True)
    metal_lb_config_from_file_help_text = '\nPath of the YAML/JSON file that contains the MetalLB configurations.\n\nExamples:\n\n  metalLBConfig:\n    addressPools:\n    - pool: lb-test-ip\n      addresses:\n      - 10.251.133.79/32\n      - 10.251.133.80/32\n      avoidBuggyIPs: True\n      manualAssign: False\n    - pool: ingress-ip\n      addresses:\n      - 10.251.133.70/32\n      avoidBuggyIPs: False\n      manualAssign: True\n\nList of supported fields in `metalLBConfig`\n\nKEY           | VALUE                     | NOTE\n--------------|---------------------------|------------------\naddressPools  | one or more addressPools  | required, mutable\n\nList of supported fields in `addressPools`\n\nKEY           | VALUE                 | NOTE\n--------------|-----------------------|---------------------------\npool          | string                | required, mutable\naddresses     | one or more IP ranges | required, mutable\navoidBuggyIPs | bool                  | optional, mutable, defaults to False\nmanualAssign  | bool                  | optional, mutable, defaults to False\n\n'
    metal_lb_config_mutex_group.add_argument('--metal-lb-config-from-file', help=metal_lb_config_from_file_help_text, type=arg_parsers.YAMLFileContents(), hidden=True)
    metal_lb_config_address_pools_help_text = "\nMetalLB load balancer configurations.\n\nExamples:\n\nTo specify MetalLB load balancer configurations for two address pools `pool1` and `pool2`,\n\n```\n$ gcloud {command}\n    --metal-lb-config-address-pools 'pool=pool1,avoid-buggy-ips=True,manual-assign=True,addresses=192.168.1.1/32;192.168.1.2-192.168.1.3'\n    --metal-lb-config-address-pools 'pool=pool2,avoid-buggy-ips=False,manual-assign=False,addresses=192.168.2.1/32;192.168.2.2-192.168.2.3'\n```\n\nUse quote around the flag value to escape semicolon in the terminal.\n"
    metal_lb_config_mutex_group.add_argument('--metal-lb-config-address-pools', help=metal_lb_config_address_pools_help_text, action='append', type=arg_parsers.ArgDict(spec={'pool': str, 'avoid-buggy-ips': arg_parsers.ArgBoolean(), 'manual-assign': arg_parsers.ArgBoolean(), 'addresses': arg_parsers.ArgList(custom_delim_char=';')}, required_keys=['pool', 'addresses']))