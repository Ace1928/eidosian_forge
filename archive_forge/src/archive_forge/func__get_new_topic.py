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
def _get_new_topic(self):
    random.seed()
    topics = random.sample(self.topic_list, self.num_topics - 1)
    topics.append(NO_TOPIC)
    letters = list(string.ascii_uppercase)[:self.num_topics]
    topic_list = {x: y for x, y in zip(letters, topics)}
    topic_text = '\n'.join(['{}: {}'.format(k, v) for k, v in topic_list.items()])
    done = False
    while not done:
        self.human_agent.observe({'text': '\nPlease choose one of the following topics by typing A, B, C, ..., etc. : \n\n{}\n'.format(topic_text)})
        topic_act = self.human_agent.act()
        choice = topic_act['text'][0].upper()
        if choice in topic_list:
            done = True
        else:
            self.human_agent.observe({'text': 'Invalid response, please try again.'})
    chosen_topic = topic_list[choice]
    print('[ Your chosen topic is: {} ]'.format(chosen_topic))
    return chosen_topic