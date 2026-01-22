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
def _generate_topic_models_worker(ensemble, num_models, random_states, pipe):
    """Wrapper for _generate_topic_models to write the results into a pipe.

    This is intended to be used inside a subprocess."""
    logger.info(f'spawned worker to generate {num_models} topic models')
    _generate_topic_models(ensemble=ensemble, num_models=num_models, random_states=random_states)
    if ensemble.memory_friendly_ttda:
        pipe.send(ensemble.ttda)
    else:
        pipe.send(ensemble.tms)
    pipe.close()