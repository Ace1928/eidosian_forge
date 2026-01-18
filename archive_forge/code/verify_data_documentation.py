from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.utils.misc import TimeLogger, warn_once
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging

Verify data doesn't have basic mistakes, like empty text fields or empty label
candidates.

Examples
--------

.. code-block:: shell

  parlai verify_data -t convai2 -dt train:ordered
