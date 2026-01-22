from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.strings import colorize
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import random

Basic example which iterates through the tasks specified and prints them out. Used for
verification of data loading and iteration.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:

Examples
--------

.. code-block:: shell

  parlai display_data -t babi:task1k:1
