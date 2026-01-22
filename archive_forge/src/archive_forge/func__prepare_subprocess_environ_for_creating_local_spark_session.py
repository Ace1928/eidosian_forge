import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _prepare_subprocess_environ_for_creating_local_spark_session():
    from mlflow.utils.databricks_utils import is_in_databricks_runtime
    if is_in_databricks_runtime():
        os.environ['SPARK_DIST_CLASSPATH'] = '/databricks/jars/*'
    os.environ.pop('PYSPARK_GATEWAY_PORT', None)
    os.environ.pop('PYSPARK_GATEWAY_SECRET', None)