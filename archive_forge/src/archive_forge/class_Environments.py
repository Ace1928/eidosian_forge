from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Environments(base.Group):
    """Create and manage Cloud Composer environments.

  The {command} command group lets you create Cloud Composer environments
  containing an Apache Airflow setup. Additionally, the command group supports
  environment updates including varying number of machines used to run Airflow,
  setting Airflow configs, or installing Python dependencies used in Airflow
  DAGs. The command group can also be used to delete Composer environments.
  """