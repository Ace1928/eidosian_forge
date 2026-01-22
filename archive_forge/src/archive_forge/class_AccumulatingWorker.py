import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class AccumulatingWorker(mp.Process):
    """Accumulate stats from texts fed in from queue."""

    def __init__(self, input_q, output_q, accumulator, window_size):
        super(AccumulatingWorker, self).__init__()
        self.input_q = input_q
        self.output_q = output_q
        self.accumulator = accumulator
        self.accumulator.log_every = sys.maxsize
        self.window_size = window_size

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            logger.info('%s interrupted after processing %d documents', self.__class__.__name__, self.accumulator.num_docs)
        except Exception:
            logger.exception('worker encountered unexpected exception')
        finally:
            self.reply_to_master()

    def _run(self):
        batch_num = -1
        n_docs = 0
        while True:
            batch_num += 1
            docs = self.input_q.get(block=True)
            if docs is None:
                logger.debug('observed sentinel value; terminating')
                break
            self.accumulator.partial_accumulate(docs, self.window_size)
            n_docs += len(docs)
            logger.debug('completed batch %d; %d documents processed (%d virtual)', batch_num, n_docs, self.accumulator.num_docs)
        logger.debug('finished all batches; %d documents processed (%d virtual)', n_docs, self.accumulator.num_docs)

    def reply_to_master(self):
        logger.info('serializing accumulator to return to master...')
        self.output_q.put(self.accumulator, block=False)
        logger.info('accumulator serialized')