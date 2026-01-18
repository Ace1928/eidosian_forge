import queue
import threading
from ray.util.timer import _Timer
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.metrics.window_stat import WindowStat
Background thread that updates the local model from replay data.

    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    