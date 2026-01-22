import logging
import os
from typing import Optional
def _try_get_delta_table_latest_version_from_path(path: str) -> Optional[int]:
    """Gets the latest version of the Delta table located at the specified path.

    Args:
        path: The path to the Delta table.

    Returns:
        The version of the Delta table, or None if it cannot be resolved (e.g. because the
        Delta core library is not installed or the specified path does not refer to a Delta
        table).

    """
    from pyspark.sql import SparkSession
    try:
        spark = SparkSession.builder.getOrCreate()
        j_delta_table = spark._jvm.io.delta.tables.DeltaTable.forPath(spark._jsparkSession, path)
        return _get_delta_table_latest_version(j_delta_table)
    except Exception as e:
        _logger.warning("Failed to obtain version information for Delta table at path '%s'. Version information may not be included in the dataset source for MLflow Tracking. Exception: %s", path, e)