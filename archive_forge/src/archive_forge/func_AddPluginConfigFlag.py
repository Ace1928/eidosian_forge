from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.service_extensions import util
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddPluginConfigFlag(parser):
    """Adds flags defining plugin config."""
    plugin_config_group = parser.add_group(mutex=True, required=False, help='Configuration for the plugin, provided at runtime by the\n              `on_configure` function (Rust Proxy-Wasm SDK) or the\n              `onConfigure` method (C++ Proxy-Wasm SDK).')
    plugin_config_group.add_argument('--plugin-config', required=False, help='Plugin runtime configuration in the textual format.')
    plugin_config_group.add_argument('--plugin-config-file', required=False, type=arg_parsers.FileContents(binary=True), help='Path to a local file containing the plugin runtime\n              configuration.')
    plugin_config_group.add_argument('--plugin-config-uri', required=False, help="URI of the container image containing the plugin's runtime\n              configuration, stored in the Artifact Registry.")