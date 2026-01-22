from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.dict import DictionaryAgent
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging

Count and display statistics of the data.

Examples
--------

.. code-block:: shell

  parlai data_stats -t convai2 -dt train:ordered
