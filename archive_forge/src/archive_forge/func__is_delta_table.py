import logging
import os
from typing import Optional
def _is_delta_table(table_name: str) -> bool:
    """Checks if a Delta table exists with the specified table name.

    Returns:
        True if a Delta table exists with the specified table name. False otherwise.

    """
    from pyspark.sql import SparkSession
    from pyspark.sql.utils import AnalysisException
    spark = SparkSession.builder.getOrCreate()
    try:
        spark.sql(f'DESCRIBE DETAIL {table_name}').filter("format = 'delta'").count()
        return True
    except AnalysisException:
        return False