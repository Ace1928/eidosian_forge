import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _get_spark_scala_version_child_proc_target(result_queue):
    from pyspark.sql import SparkSession
    _prepare_subprocess_environ_for_creating_local_spark_session()
    with SparkSession.builder.master('local[1]').getOrCreate() as spark_session:
        scala_version = _get_spark_scala_version_from_spark_session(spark_session)
        result_queue.put(scala_version)