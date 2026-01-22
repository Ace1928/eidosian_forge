import logging
from typing import List, Optional
from ray.data._internal.util import _check_import
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _validate_dataset_table_exist(self, project_id: str, dataset: str) -> None:
    from google.api_core import exceptions
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id)
    dataset_id = dataset.split('.')[0]
    try:
        client.get_dataset(dataset_id)
    except exceptions.NotFound:
        raise ValueError('Dataset {} is not found. Please ensure that it exists.'.format(dataset_id))
    try:
        client.get_table(dataset)
    except exceptions.NotFound:
        raise ValueError('Table {} is not found. Please ensure that it exists.'.format(dataset))