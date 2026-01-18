from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.scripts.train_model import setup_args as train_args
from parlai.scripts.train_model import TrainLoop
import parlai.utils.logging as logging
import cProfile
import io
import pdb
import pstats

Run the python or pytorch profiler and prints the results.

Examples
--------

To make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:

.. code-block:: shell

  parlai profile_train -t babi:task1k:1 -m seq2seq -e 0.1 --dict-file /tmp/dict
