from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.command_lib.logs import stream
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_prep
from googlecloudsdk.command_lib.ml_engine import log_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def SubmitPrediction(jobs_client, job, model_dir=None, model=None, version=None, input_paths=None, data_format=None, output_path=None, region=None, runtime_version=None, max_worker_count=None, batch_size=None, signature_name=None, labels=None, accelerator_count=None, accelerator_type=None):
    """Submit a prediction job."""
    _ValidateSubmitPredictionArgs(model_dir, version)
    project_ref = resources.REGISTRY.Parse(properties.VALUES.core.project.Get(required=True), collection='ml.projects')
    job = jobs_client.BuildBatchPredictionJob(job_name=job, model_dir=model_dir, model_name=model, version_name=version, input_paths=input_paths, data_format=data_format, output_path=output_path, region=region, runtime_version=runtime_version, max_worker_count=max_worker_count, batch_size=batch_size, signature_name=signature_name, labels=labels, accelerator_count=accelerator_count, accelerator_type=_ACCELERATOR_MAP.GetEnumForChoice(accelerator_type))
    PrintSubmitFollowUp(job.jobId, print_follow_up_message=True)
    return jobs_client.Create(project_ref, job)