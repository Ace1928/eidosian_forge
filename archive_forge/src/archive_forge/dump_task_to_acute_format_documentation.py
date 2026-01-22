from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.conversations import Conversations
from parlai.utils.misc import TimeLogger
import random
import tempfile

    Dump task data to ACUTE-Eval.
    