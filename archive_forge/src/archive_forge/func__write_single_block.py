import logging
import os
import tempfile
import time
import uuid
from typing import Any, Iterable
import pyarrow.parquet as pq
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.util import _check_import
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.datasink import Datasink
def _write_single_block(block: Block, project_id: str, dataset: str) -> None:
    from google.api_core import exceptions
    from google.cloud import bigquery
    block = BlockAccessor.for_block(block).to_arrow()
    client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(autodetect=True)
    job_config.source_format = bigquery.SourceFormat.PARQUET
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    with tempfile.TemporaryDirectory() as temp_dir:
        fp = os.path.join(temp_dir, f'block_{uuid.uuid4()}.parquet')
        pq.write_table(block, fp, compression='SNAPPY')
        retry_cnt = 0
        while retry_cnt <= self.max_retry_cnt:
            with open(fp, 'rb') as source_file:
                job = client.load_table_from_file(source_file, dataset, job_config=job_config)
            try:
                logger.info(job.result())
                break
            except exceptions.Forbidden as e:
                retry_cnt += 1
                if retry_cnt > self.max_retry_cnt:
                    break
                logger.info('A block write encountered a rate limit exceeded error' + f' {retry_cnt} time(s). Sleeping to try again.')
                logging.debug(e)
                time.sleep(RATE_LIMIT_EXCEEDED_SLEEP_TIME)
        if retry_cnt > self.max_retry_cnt:
            logger.info(f'Maximum ({self.max_retry_cnt}) retry count exceeded. Ray' + ' will attempt to retry the block write via fault tolerance.')
            raise RuntimeError(f'Write failed due to {retry_cnt}' + ' repeated API rate limit exceeded responses. Consider' + ' specifiying the max_retry_cnt kwarg with a higher value.')