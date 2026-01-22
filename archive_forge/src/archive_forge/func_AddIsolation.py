from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def AddIsolation(self, allow_data_boost=False):
    """Add argument for isolating this app profile's traffic to parser."""
    isolation_group = self.parser.add_mutually_exclusive_group()
    standard_isolation_group = isolation_group.add_group('Standard Isolation')
    choices = {'PRIORITY_LOW': 'Requests are treated with low priority.', 'PRIORITY_MEDIUM': 'Requests are treated with medium priority.', 'PRIORITY_HIGH': 'Requests are treated with high priority.'}
    standard_isolation_group.add_argument('--priority', type=lambda x: x.replace('-', '_').upper(), choices=choices, default=None, help='Specify the request priority under Standard Isolation. Passing this option implies Standard Isolation, e.g. the `--standard` option. If not specified, the app profile uses Standard Isolation with PRIORITY_HIGH by default. Specifying request priority on an app profile that has Data Boost Read-Only Isolation enabled will change the isolation to Standard and use the specified priority, which may cause unexpected behavior for running applications.' if allow_data_boost else 'Specify the request priority. If not specified, the app profile uses PRIORITY_HIGH by default.', required=True)
    if allow_data_boost:
        standard_isolation_group.add_argument('--standard', action='store_true', default=False, help='Use Standard Isolation, rather than Data Boost Read-only Isolation. If specified, `--priority` is required.')
        data_boost_isolation_group = isolation_group.add_group('Data Boost Read-only Isolation')
        data_boost_isolation_group.add_argument('--data-boost', action='store_true', default=False, help='Use Data Boost Read-only Isolation, rather than Standard Isolation. If specified, --data-boost-compute-billing-owner is required. Specifying Data Boost Read-only Isolation on an app profile which has Standard Isolation enabled may cause unexpected behavior for running applications.', required=True)
        compute_billing_choices = {'HOST_PAYS': 'Compute Billing should be accounted towards the host Cloud Project (containing the targeted Bigtable Instance / Table).'}
        data_boost_isolation_group.add_argument('--data-boost-compute-billing-owner', type=lambda x: x.upper(), choices=compute_billing_choices, default=None, help='Specify the Data Boost Compute Billing Owner, required if --data-boost is passed.', required=True)
    return self