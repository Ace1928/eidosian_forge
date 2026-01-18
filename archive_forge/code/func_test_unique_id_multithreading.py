import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_unique_id_multithreading():
    """Create a bunch of threads and assert they all get unique IDs"""
    list_of_ids = []

    def get_model_id(id_list, index):
        id_list.append(create_model(name=f'worker{index}').id)
    counter = 0
    while len(list_of_ids) < 1000:
        workers = []
        for i in range(50):
            w = threading.Thread(target=get_model_id, args=(list_of_ids, counter))
            workers.append(w)
            counter += 1
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    assert len(list_of_ids) == len(list(set(list_of_ids)))