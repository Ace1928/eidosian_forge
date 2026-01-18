from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
Parses the name of a pipelineRun/taskRun.

  Args:
    pattern:
      "projects/{project}/locations/{location}/pipelineRuns/{pipeline_run}"
      "projects/{project}/locations/{location}/taskRuns/{task_run}"
    primitive_type: string

  Returns:
    name: string
  