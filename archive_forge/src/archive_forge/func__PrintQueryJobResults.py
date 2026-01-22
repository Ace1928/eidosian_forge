from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import sys
from typing import Optional
from absl import app
from absl import flags
from pyglib import appcommands
from clients import bigquery_client
from clients import bigquery_client_extended
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from frontend import utils_data_transfer
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
def _PrintQueryJobResults(self, client: bigquery_client_extended.BigqueryClientExtended, job) -> None:
    """Prints the results of a successful query job.

    This function is invoked only for successful jobs.  Output is printed to
    stdout.  Depending on flags, the output is printed in either free-form or
    json style.

    Args:
      client: Bigquery client object
      job: json of the job, expressed as a dictionary
    """
    if job['statistics']['query']['statementType'] == 'SCRIPT':
        self._PrintScriptJobResults(client, job)
    else:
        self.PrintNonScriptQueryJobResults(client, job)