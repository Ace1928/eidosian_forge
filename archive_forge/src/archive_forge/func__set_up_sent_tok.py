from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def _set_up_sent_tok(self):
    try:
        import nltk
    except ImportError:
        raise ImportError('Please install nltk (e.g. pip install nltk).')
    st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
    try:
        self.sent_tok = nltk.data.load(st_path)
    except LookupError:
        nltk.download('punkt')
        self.sent_tok = nltk.data.load(st_path)