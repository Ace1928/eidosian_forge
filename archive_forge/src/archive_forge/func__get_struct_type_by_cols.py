import re
from functools import reduce
from typing import Set, Union
from pyspark.ml.base import Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import types as t
def _get_struct_type_by_cols(input_fields: Set[str], df_schema: t.StructType) -> t.StructType:
    """
    Args:
        input_fields: A set of input columns to be
                    intersected with the input dataset's columns.
        df_schema: A Spark dataframe schema to compare input_fields

    Returns:
        A StructType from the intersection of given columns and
        the columns present in the training dataset
    """
    if len(input_fields) > 0:
        return t.StructType([_field for _field in df_schema.fields if _field.name in input_fields])
    return []