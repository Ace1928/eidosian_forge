import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def _calculate_assymetric_distance_matrix_multiproc(workers, entire_ttda, masking_method, masking_threshold):
    processes = []
    pipes = []
    ttdas_sent = 0
    for i in range(workers):
        try:
            parent_conn, child_conn = Pipe()
            n_ttdas = 0
            if i == workers - 1:
                n_ttdas = len(entire_ttda) - ttdas_sent
            else:
                n_ttdas = int((len(entire_ttda) - ttdas_sent) / (workers - i))
            args = (i, entire_ttda, ttdas_sent, n_ttdas, masking_method, masking_threshold, child_conn)
            process = Process(target=_asymmetric_distance_matrix_worker, args=args)
            ttdas_sent += n_ttdas
            processes.append(process)
            pipes.append((parent_conn, child_conn))
            process.start()
        except ProcessError:
            logger.error(f'could not start process {i}')
            _teardown(pipes, processes)
            raise
    distances = []
    for parent_conn, _ in pipes:
        worker_id, distance_chunk = parent_conn.recv()
        parent_conn.close()
        distances.append(distance_chunk)
    for process in processes:
        process.terminate()
    return np.concatenate(distances)