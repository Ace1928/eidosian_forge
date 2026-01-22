import abc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.artifacts import DataframeArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.steps.ingest.datasets import (
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.utils.file_utils import read_parquet_as_pandas_df
def _build_step_card(self, ingested_dataset_profile: str, ingested_rows: int, schema: Dict, data_preview: pd.DataFrame=None, dataset_src_location: Optional[str]=None, dataset_sql: Optional[str]=None) -> BaseCard:
    """
        Constructs a step card instance corresponding to the current ingest step state.

        Args:
            ingested_dataset_path: The local filesystem path to the ingested parquet dataset file.
            dataset_src_location: The source location of the dataset (e.g. '/tmp/myfile.parquet',
                's3://mybucket/mypath', ...), if the dataset is a location-based dataset. Either
                ``dataset_src_location`` or ``dataset_sql`` must be specified.
            dataset_sql: The Spark SQL query string that defines the dataset
                (e.g. 'SELECT * FROM my_spark_table'), if the dataset is a Spark SQL dataset. Either
                ``dataset_src_location`` or ``dataset_sql`` must be specified.

        Returns:
            An BaseCard instance corresponding to the current ingest step state.

        """
    if dataset_src_location is None and dataset_sql is None:
        raise MlflowException(message='Failed to build step card because neither a dataset location nor a dataset Spark SQL query were specified', error_code=INVALID_PARAMETER_VALUE)
    card = BaseCard(self.recipe_name, self.name)
    if not self.skip_data_profiling:
        card.add_tab('Data Profile', '{{PROFILE}}').add_pandas_profile('PROFILE', ingested_dataset_profile)
    schema_html = BaseCard.render_table(schema['fields'])
    card.add_tab('Data Schema', '{{SCHEMA}}').add_html('SCHEMA', schema_html)
    if data_preview is not None:
        card.add_tab('Data Preview', '{{DATA_PREVIEW}}').add_html('DATA_PREVIEW', BaseCard.render_table(data_preview))
    card.add_tab('Run Summary', '{{ INGESTED_ROWS }}' + '{{ DATA_SOURCE }}' + '{{ EXE_DURATION }}' + '{{ LAST_UPDATE_TIME }}').add_markdown(name='INGESTED_ROWS', markdown=f'**Number of rows ingested:** `{ingested_rows}`').add_markdown(name='DATA_SOURCE', markdown=f'**Dataset source location:** `{dataset_src_location}`' if dataset_src_location is not None else f'**Dataset SQL:** `{dataset_sql}`')
    return card