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
def AddWasmPluginVersionArgs(parser, version_message):
    wasm_plugin_version_group = parser.add_group(mutex=False, required=False)
    AddVersionFlag(wasm_plugin_version_group, version_message)
    AddImageFlag(wasm_plugin_version_group)
    AddPluginConfigFlag(wasm_plugin_version_group)