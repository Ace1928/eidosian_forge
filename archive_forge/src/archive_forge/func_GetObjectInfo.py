from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def GetObjectInfo(self, reference):
    """Get all data returned by the server about a specific object."""
    if isinstance(reference, bq_id_utils.ApiClientHelper.ProjectReference):
        max_project_results = 1000
        projects = self.ListProjects(max_results=max_project_results)
        for project in projects:
            if bq_processor_utils.ConstructObjectReference(project) == reference:
                project['kind'] = 'bigquery#project'
                return project
        if len(projects) >= max_project_results:
            raise bq_error.BigqueryError('Number of projects found exceeded limit, please instead run gcloud projects describe %s' % (reference,))
        raise bq_error.BigqueryNotFoundError('Unknown %r' % (reference,), {'reason': 'notFound'}, [])
    if isinstance(reference, bq_id_utils.ApiClientHelper.JobReference):
        return self.apiclient.jobs().get(**dict(reference)).execute()
    elif isinstance(reference, bq_id_utils.ApiClientHelper.DatasetReference):
        return self.apiclient.datasets().get(**dict(reference)).execute()
    elif isinstance(reference, bq_id_utils.ApiClientHelper.TableReference):
        return self.apiclient.tables().get(**dict(reference)).execute()
    elif isinstance(reference, bq_id_utils.ApiClientHelper.ModelReference):
        return self.GetModelsApiClient().models().get(projectId=reference.projectId, datasetId=reference.datasetId, modelId=reference.modelId).execute()
    elif isinstance(reference, bq_id_utils.ApiClientHelper.RoutineReference):
        return self.GetRoutinesApiClient().routines().get(projectId=reference.projectId, datasetId=reference.datasetId, routineId=reference.routineId).execute()
    else:
        raise TypeError('Type of reference must be one of: ProjectReference, JobReference, DatasetReference, or TableReference')