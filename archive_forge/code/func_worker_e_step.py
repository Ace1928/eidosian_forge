import logging
import queue
from multiprocessing import Pool, Queue, cpu_count
import numpy as np
from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState
def worker_e_step(input_queue, result_queue, worker_lda):
    """Perform E-step for each job.

    Parameters
    ----------
    input_queue : queue of (int, list of (int, float), :class:`~gensim.models.lda_worker.Worker`)
        Each element is a job characterized by its ID, the corpus chunk to be processed in BOW format and the worker
        responsible for processing it.
    result_queue : queue of :class:`~gensim.models.ldamodel.LdaState`
        After the worker finished the job, the state of the resulting (trained) worker model is appended to this queue.
    worker_lda : :class:`~gensim.models.ldamulticore.LdaMulticore`
        LDA instance which performed e step
    """
    logger.debug('worker process entering E-step loop')
    while True:
        logger.debug('getting a new job')
        chunk_no, chunk, w_state = input_queue.get()
        logger.debug('processing chunk #%i of %i documents', chunk_no, len(chunk))
        worker_lda.state = w_state
        worker_lda.sync_state()
        worker_lda.state.reset()
        worker_lda.do_estep(chunk)
        del chunk
        logger.debug('processed chunk, queuing the result')
        result_queue.put(worker_lda.state)
        worker_lda.state = None
        logger.debug('result put')