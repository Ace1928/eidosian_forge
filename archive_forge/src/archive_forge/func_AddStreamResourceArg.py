from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddStreamResourceArg(parser, verb, release_track, required=True):
    """Add resource arguments for creating/updating a stream.

  Args:
    parser: argparse.ArgumentParser, the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    release_track: base.ReleaseTrack, some arguments are added based on the
        command release track.
    required: bool, if True, means that a flag is required.
  """
    source_parser = parser.add_group(required=required)
    source_config_parser_group = source_parser.add_group(required=required, mutex=True)
    source_config_parser_group.add_argument('--oracle-source-config', help=_ORACLE_SOURCE_CONFIG_HELP_TEXT_BETA if release_track == base.ReleaseTrack.BETA else _ORACLE_SOURCE_CONFIG_HELP_TEXT)
    source_config_parser_group.add_argument('--mysql-source-config', help=_MYSQL_SOURCE_CONFIG_HELP_TEXT_BETA if release_track == base.ReleaseTrack.BETA else _MYSQL_SOURCE_CONFIG_HELP_TEXT)
    source_config_parser_group.add_argument('--postgresql-source-config', help=_POSTGRESQL_UPDATE_SOURCE_CONFIG_HELP_TEXT if verb == 'update' else _POSTGRESQL_CREATE_SOURCE_CONFIG_HELP_TEXT)
    destination_parser = parser.add_group(required=required)
    destination_config_parser_group = destination_parser.add_group(required=required, mutex=True)
    destination_config_parser_group.add_argument('--gcs-destination-config', help='      Path to a YAML (or JSON) file containing the configuration for Google Cloud Storage Destination Config.\n\n      The JSON file is formatted as follows:\n\n      ```\n       {\n       "path": "some/path",\n       "fileRotationMb":5,\n       "fileRotationInterval":"15s",\n       "avroFileFormat": {}\n       }\n      ```\n        ')
    destination_config_parser_group.add_argument('--bigquery-destination-config', help='      Path to a YAML (or JSON) file containing the configuration for Google BigQuery Destination Config.\n\n      The JSON file is formatted as follows:\n\n      ```\n      {\n        "sourceHierarchyDatasets": {\n          "datasetTemplate": {\n            "location": "us-central1",\n            "datasetIdPrefix": "my_prefix",\n            "kmsKeyName": "projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{cryptoKey}"\n          }\n        },\n        "dataFreshness": "3600s"\n      }\n      ```\n        ')
    source_field = 'source'
    destination_field = 'destination'
    if release_track == base.ReleaseTrack.BETA:
        source_field = 'source-name'
        destination_field = 'destination-name'
    resource_specs = [presentation_specs.ResourcePresentationSpec('stream', GetStreamResourceSpec(), 'The stream to {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--%s' % source_field, GetConnectionProfileResourceSpec(), 'Resource ID of the source connection profile.', required=required, flag_name_overrides={'location': ''}, group=source_parser), presentation_specs.ResourcePresentationSpec('--%s' % destination_field, GetConnectionProfileResourceSpec(), 'Resource ID of the destination connection profile.', required=required, flag_name_overrides={'location': ''}, group=destination_parser)]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--%s.location' % source_field: ['--location'], '--%s.location' % destination_field: ['--location']}).AddToParser(parser)