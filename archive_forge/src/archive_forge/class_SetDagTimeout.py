from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import workflow_templates
from googlecloudsdk.core import log
import six
class SetDagTimeout(base.CreateCommand):
    """Set DAG timeout on a workflow template."""
    detailed_help = DETAILED_HELP

    @classmethod
    def Args(cls, parser):
        dataproc = dp.Dataproc(cls.ReleaseTrack())
        workflow_templates.AddDagTimeoutFlag(parser, True)
        flags.AddTemplateResourceArg(parser, 'set the DAG timeout on', dataproc.api_version)

    def Run(self, args):
        dataproc = dp.Dataproc(self.ReleaseTrack())
        template_ref = args.CONCEPTS.template.Parse()
        workflow_template = dataproc.GetRegionsWorkflowTemplate(template_ref, args.version)
        workflow_template.dagTimeout = six.text_type(args.dag_timeout) + 's'
        response = dataproc.client.projects_regions_workflowTemplates.Update(workflow_template)
        log.status.Print('Set a DAG timeout of {0} on {1}.'.format(workflow_template.dagTimeout, template_ref.Name()))
        return response