import re
from functools import reduce
from typing import Set, Union
from pyspark.ml.base import Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import types as t
def _do_pipeline_transform(df: DataFrame, transformer: Union[Transformer, PipelineModel]):
    """
    A util method that runs transform on a pipeline model/transformer

    Args:
        df: a spark dataframe

    Returns:
        output transformed dataframe using pipeline model/transformer
    """
    return transformer.transform(df)