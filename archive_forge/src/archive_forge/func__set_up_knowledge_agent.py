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
def _set_up_knowledge_agent(self, add_token_knowledge=False):
    from parlai.core.params import ParlaiParser
    parser = ParlaiParser(False, False)
    KnowledgeRetrieverAgent.add_cmdline_args(parser)
    parser.set_params(model='projects:wizard_of_wikipedia:knowledge_retriever', add_token_knowledge=add_token_knowledge)
    knowledge_opt = parser.parse_args([])
    self.knowledge_agent = create_agent(knowledge_opt)