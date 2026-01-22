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
def AddWasmPluginResource(parser, api_version, message):
    wasm_plugin_data = yaml_data.ResourceYAMLData.FromPath('service_extensions.wasmPlugin')
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('wasm_plugin', concepts.ResourceSpec.FromYaml(wasm_plugin_data.GetData(), api_version=api_version), message, required=True)]).AddToParser(parser)