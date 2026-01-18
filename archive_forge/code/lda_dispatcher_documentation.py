import argparse
import os
import sys
import logging
import threading
import time
from queue import Queue
import Pyro4
from gensim import utils
from gensim.models.lda_worker import LDA_WORKER_PREFIX
Terminate all registered workers and then the dispatcher.