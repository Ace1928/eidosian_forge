from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def tfidf_retriever_act(self, history):
    """
        Combines and formats texts retrieved by the TFIDF retriever for the chosen
        topic, the last thing the wizard said, and the last thing the apprentice said.
        """
    chosen_topic_txts = None
    if self.retriever_history.get('chosen_topic'):
        chosen_topic_txts, chosen_topic_txts_no_title = self.get_chosen_topic_passages(self.retriever_history['chosen_topic'])
    apprentice_txts = None
    if self.retriever_history.get('apprentice'):
        apprentice_act = {'text': self.retriever_history['apprentice'], 'episode_done': True}
        self.retriever.observe(apprentice_act)
        apprentice_txts, apprentice_txts_no_title = self.get_passages(self.retriever.act())
    wizard_txts = None
    if self.retriever_history.get('wizard'):
        wizard_act = {'text': self.retriever_history['wizard'], 'episode_done': True}
        self.retriever.observe(wizard_act)
        wizard_txts, wizard_txts_no_title = self.get_passages(self.retriever.act())
    combined_txt = ''
    combined_txt_no_title = ''
    if chosen_topic_txts:
        combined_txt += chosen_topic_txts
        combined_txt_no_title += chosen_topic_txts_no_title
    if wizard_txts:
        combined_txt += '\n' + wizard_txts
        combined_txt_no_title += '\n' + wizard_txts_no_title
    if apprentice_txts:
        combined_txt += '\n' + apprentice_txts
        combined_txt_no_title += '\n' + apprentice_txts_no_title
    return (combined_txt, combined_txt_no_title)