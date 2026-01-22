import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.utils.async_logging.run_batch import RunBatch
from mlflow.utils.async_logging.run_operations import RunOperations
def _at_exit_callback(self) -> None:
    """Callback function to be executed when the program is exiting.

        Stops the data processing thread and waits for the queue to be drained. Finally, shuts down
        the thread pools used for data logging and batch processing status check.
        """
    try:
        self._stop_data_logging_thread_event.set()
        self._batch_logging_thread.join()
        self._batch_logging_worker_threadpool.shutdown(wait=True)
        self._batch_status_check_threadpool.shutdown(wait=True)
    except Exception as e:
        _logger.error(f'Encountered error while trying to finish logging: {e}')