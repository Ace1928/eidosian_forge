from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.printers import export_printer
from googlecloudsdk.command_lib.run.printers import job_printer
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.resource import resource_printer
@staticmethod
def CommonArgs(parser):
    task_presentation = presentation_specs.ResourcePresentationSpec('TASK', resource_args.GetTaskResourceSpec(), 'Task to describe.', required=True, prefixes=False)
    concept_parsers.ConceptParser([task_presentation]).AddToParser(parser)
    resource_printer.RegisterFormatter(job_printer.TASK_PRINTER_FORMAT, job_printer.TaskPrinter, hidden=True)
    parser.display_info.AddFormat(job_printer.TASK_PRINTER_FORMAT)