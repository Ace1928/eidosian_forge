from copy import deepcopy
import json
import random
import os
import string
from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.utils.misc import warn_once
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
def _load_topics(self, opt):
    topics_path = os.path.join(opt['datapath'], 'wizard_of_wikipedia', 'topic_splits.json')
    datatype = opt['datatype'].split(':')[0]
    self.topic_list = json.load(open(topics_path, 'rb'))[datatype]