from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
def AddAspectFlags(parser: parser_arguments.ArgumentInterceptor, *, update_aspects_name: str | None='update-aspects', remove_aspects_name: str | None='remove-aspects', required: bool=False):
    """Adds flags for updating and removing Aspects.

  Args:
    parser: The arg parser to add flags to.
    update_aspects_name: Name of the flag to add for updating Aspects or None if
      no flag should be added.
    remove_aspects_name: Name of the flag to add for removing Aspects or None if
      no flag should be added.
    required: If True, then flags will be marked as required.
  """
    combination_help_text = ''
    if update_aspects_name is not None and remove_aspects_name is not None:
        combination_help_text = f'\n\n        If both `--{update_aspects_name}` and `--{remove_aspects_name}` flags\n        are specified, and the same aspect key is used in both flags, then\n        `--{update_aspects_name}` takes precedence, and such an aspect will be\n        updated and not removed.\n    '
    if update_aspects_name is not None:
        parser.add_argument(f'--{update_aspects_name}', help='\n        Path to a YAML or JSON file containing Aspects to add or update.\n\n        When this flag is specified, only Aspects referenced in the file are\n        going to be added or updated. Specifying this flag does not remove any\n        Aspects from the entry. In other words, specifying this flag will not\n        lead to a full replacement of Aspects with a contents of the provided\n        file.\n\n        Content of the file contains a map, where keys are in the format\n        ``ASPECT_TYPE@PATH\'\', or just ``ASPECT_TYPE\'\', if the Aspect is attached\n        to an entry itself rather than to a specific column defined in the\n        schema.\n\n        Values in the map represent Aspect\'s content, which must conform to a\n        template defined for a given ``ASPECT_TYPE\'\'. Each Aspect will be replaced\n        fully by the provided content. That means data in the Aspect will be\n        replaced and not merged with existing contents of that Aspect in the Entry.\n\n        ``ASPECT_TYPE\'\' is expected to be in a format\n        ``PROJECT_ID.LOCATION.ASPECT_TYPE_ID\'\'.\n\n        ``PATH\'\' can be either empty (which means a \'root\' path, such that Aspect\n        is attached to the entry itself) or point to a specific column defined\n        in the schema. For example: `Schema.some_column`.\n\n        Example YAML format:\n\n        ```\n          project-id1.us-central1.my-aspect-type1:\n            data:\n              aspectField1: someValue\n              aspectField2: someOtherValue\n          project-id2.us-central1.my-aspect-type2@Schema.column1:\n            data:\n              aspectField3: someValue3\n        ```\n\n        Example JSON format:\n\n        ```\n          {\n            "project-id1.us-central1.my-aspect-type1": {\n              "data": {\n                "aspectField1": "someValue",\n                "aspectField2": "someOtherValue"\n              }\n            },\n            "project-id2.us-central1.my-aspect-type2@Schema.column1": {\n              "data": {\n                "aspectField3": "someValue3"\n              }\n            }\n          }\n        ```\n        ' + combination_help_text, type=dataplex_parsers.ParseAspects, metavar='YAML_OR_JSON_FILE', required=required)
    if remove_aspects_name is not None:
        parser.add_argument(f'--{remove_aspects_name}', help="\n        List of Aspect keys, identifying Aspects to remove from the entry.\n\n        Keys are in the format ``ASPECT_TYPE@PATH'', or just ``ASPECT_TYPE'', if\n        the Aspect is attached to an entry itself rather than to a specific\n        column defined in the schema.\n\n        ``ASPECT_TYPE'' is expected to be in a format\n        ``PROJECT_ID.LOCATION.ASPECT_TYPE_ID'' or a wildcard `*`, which targets\n        all aspect types.\n\n        ``PATH'' can be either empty (which means a 'root' path, such that\n        Aspect is attached to the entry itself), point to a specific column\n        defined in the schema (for example: `Schema.some_column`) or a wildcard\n        `*` (target all paths).\n\n        ``ASPECT_TYPE'' and ``PATH'' cannot be both specified as wildcards `*`." + combination_help_text, type=arg_parsers.ArgList(), metavar='ASPECT_TYPE@PATH', required=required)