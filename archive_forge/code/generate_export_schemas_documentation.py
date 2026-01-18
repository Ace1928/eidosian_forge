from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.meta.apis import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import export
from googlecloudsdk.command_lib.util.apis import registry
Generate YAML export schemas for a message in a given API.

  *gcloud* commands that have "too many" *create*/*update* command flags may
  also provide *export*/*import* commands. *export* lists the current state
  of a resource in a YAML *export* format. *import* reads export format data
  and either creates a new resource or updates an existing resource.

  An export format is an abstract YAML representation of the mutable fields of a
  populated protobuf message. Abstraction allows the export format to hide
  implementation details of some protobuf constructs like enums and
  `additionalProperties`.

  One way of describing an export format is with JSON schemas. A schema
  documents export format properties for a message in an API, and can also be
  used to validate data on import. Validation is important because users can
  modify export data before importing it again.

  This command generates [JSON schemas](json-schema.org) (in YAML format, go
  figure) for a protobuf message in an API. A separate schema files is
  generated for each nested message in the resource message.

  ## CAVEATS

  The generated schemas depend on the quality of the protobuf discovery
  docs, including proto file comment conventions that are not error checked.
  Always manually inspect schemas before using them in a release.

  ## EXAMPLES

  To generate the WorkflowTemplate schemas in the current directory from the
  dataproc v1 API:

    $ {command} WorkflowTemplate --api=dataproc --api-version=v1
  