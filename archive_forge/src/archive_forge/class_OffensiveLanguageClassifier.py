from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
import os
class OffensiveLanguageClassifier:
    """
    Load model trained to detect offensive language in the context of single- turn
    dialogue utterances.

    This model was trained to be robust to adversarial examples created by humans. See
    <http://parl.ai/projects/dialogue_safety/> for more information.
    """

    def __init__(self, shared: TShared=None, custom_model_file='zoo:dialogue_safety/single_turn/model'):
        if not shared:
            self.model = self._create_safety_model(custom_model_file)
        else:
            self.model = create_agent_from_shared(shared['model'])
        self.classes = {OK_CLASS: False, NOT_OK_CLASS: True}

    def share(self):
        shared = {'model': self.model.share()}
        return shared

    def _create_safety_model(self, custom_model_file):
        from parlai.core.params import ParlaiParser
        parser = ParlaiParser(False, False)
        TransformerClassifierAgent.add_cmdline_args(parser)
        parser.set_params(model='transformer/classifier', model_file=custom_model_file, print_scores=True)
        safety_opt = parser.parse_args([])
        return create_agent(safety_opt, requireModelExists=True)

    def contains_offensive_language(self, text):
        """
        Returns the probability that a message is safe according to the classifier.
        """
        act = {'text': text, 'episode_done': True}
        self.model.observe(act)
        response = self.model.act()['text']
        pred_class, prob = [x.split(': ')[-1] for x in response.split('\n')]
        pred_not_ok = self.classes[pred_class]
        prob = float(prob)
        return (pred_not_ok, prob)

    def __contains__(self, key):
        """
        A simple way of checking whether the model classifies an utterance as offensive.

        Returns True if the input phrase is offensive.
        """
        pred_not_ok, prob = self.contains_offensive_language(key)
        return pred_not_ok