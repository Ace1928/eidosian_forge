import logging
import queue
from multiprocessing import Pool, Queue, cpu_count
import numpy as np
from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState

            Clear the result queue, merging all intermediate results, and update the
            LDA model if necessary.

            