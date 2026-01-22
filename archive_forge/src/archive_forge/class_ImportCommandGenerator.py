from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
class ImportCommandGenerator(BaseCommandGenerator):
    """Generator for import commands."""
    command_type = yaml_command_schema.CommandType.IMPORT

    def _Generate(self):
        """Generates an import command.

    An import command has a single resource argument and an API method to call
    to get the resource. The result is from a local yaml file provided
    by the `--source` flag, or from stdout if nothing is provided.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        from googlecloudsdk.command_lib.export import util as export_util

        class Command(base.ImportCommand):
            """Import command enclosure."""

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                if self.spec.async_:
                    base.ASYNC_FLAG.AddToParser(parser)
                parser.add_argument('--source', help="\n            Path to a YAML file containing the configuration export data. The\n            YAML file must not contain any output-only fields. Alternatively, you\n            may omit this flag to read from standard input. For a schema\n            describing the export/import format, see:\n            $CLOUDSDKROOT/lib/googlecloudsdk/schemas/...\n\n            $CLOUDSDKROOT is can be obtained with the following command:\n\n              $ gcloud info --format='value(installation.sdk_root)'\n          ")

            def Run(self_, args):
                method = self.arg_generator.GetPrimaryResource(self.methods, args).method
                message_type = method.GetRequestType()
                request_field = method.request_field
                resource_message_class = message_type.field_by_name(request_field).type
                data = console_io.ReadFromFileOrStdin(args.source or '-', binary=False)
                schema_path = export_util.GetSchemaPath(method.collection.api_name, self.spec.request.api_version, resource_message_class.__name__)
                imported_resource = export_util.Import(message_type=resource_message_class, stream=data, schema_path=schema_path)
                existing_resource = None
                if self.spec.import_:
                    abort_if_equivalent = self.spec.import_.abort_if_equivalent
                    create_if_not_exists = self.spec.import_.create_if_not_exists
                    try:
                        existing_resource = self._GetExistingResource(args)
                    except apitools_exceptions.HttpError as error:
                        if error.status_code != 404 or not create_if_not_exists:
                            raise error
                        else:
                            self.spec.request = self.spec.import_.create_request
                            if self.spec.import_.no_create_async:
                                self.spec.async_ = None
                            elif self.spec.import_.create_async:
                                self.spec.async_ = self.spec.import_.create_async
                            self.InitializeGeneratorForCommand()
                    if abort_if_equivalent:
                        if imported_resource == existing_resource:
                            return log.status.Print('Request not sent for [{}]: No changes detected.'.format(imported_resource.name))
                ref, response = self._CommonRun(args, existing_message=imported_resource)
                if self.spec.async_:
                    request_string = None
                    if ref is not None:
                        request_string = 'Request issued for: [{{{}}}]'.format(yaml_command_schema_util.NAME_FORMAT_KEY)
                    response = self._HandleAsync(args, ref, response, request_string)
                return self._HandleResponse(response, args)
        return Command