from parlai.agents.transformer.transformer import TransformerRankerAgent
from .wizard_dict import WizardDictAgent
import numpy as np
import torch
class WizardTransformerRankerAgent(TransformerRankerAgent):

    @staticmethod
    def dictionary_class():
        return WizardDictAgent

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        super(WizardTransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Wizard Transformer Ranker Arguments')
        agent.add_argument('--use-knowledge', type='bool', default=True, help='use knowledge field instead of personas')
        agent.add_argument('--knowledge-dropout', type=float, default=0.7, help='dropout some knowledge during training')
        agent.add_argument('--chosen-sentence', type='bool', default=False, help='instead of using all knowledge, use goldlabel, i.e. the chosen sentence')
        agent.add_argument('--knowledge-truncate', type=int, default=50, help='truncate knowledge to this length')
        agent.add_argument('--legacy', type='bool', default=False, help='legacy model')
        argparser.set_defaults(learningrate=0.0008, eval_candidates='inline', candidates='batch', lr_factor=1, add_p1_after_newln=False, delimiter=' ')
        return agent

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        super().__init__(opt, shared)
        self.use_knowledge = opt.get('use_knowledge', False)
        if self.use_knowledge:
            self.opt['use_memories'] = True
        self.chosen_sentence = opt.get('chosen_sentence', False) and self.use_knowledge
        self.knowledge_dropout = opt.get('knowledge_dropout', 0)
        self.knowledge_truncate = opt.get('knowledge_truncate', 50)

    def _set_text_vec(self, *args, **kwargs):
        """
        Sets the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if self.opt.get('legacy') and 'text_vec' in obs:
            if obs['text_vec'][0] != self.dict[SOC_TOKEN]:
                soc_tensor = torch.LongTensor([self.dict[SOC_TOKEN]])
                obs.force_set('text_vec', torch.cat([soc_tensor, obs['text_vec']]))
        return obs

    def _vectorize_memories(self, observation):
        """
        Override abstract method from TransformerRankerAgent to use knowledge field as
        memories.
        """
        if not self.use_knowledge:
            return observation
        observation['memory_vecs'] = []
        checked = observation.get('checked_sentence', '')
        if observation.get('knowledge'):
            knowledge = observation['knowledge'].split('\n')[:-1]
        else:
            knowledge = []
        to_vectorize = []
        if checked and self.chosen_sentence:
            to_vectorize = [checked]
        elif (self.knowledge_dropout == 0 or not self.is_training) and knowledge:
            to_vectorize = knowledge
        elif knowledge:
            for line in knowledge:
                if checked and checked in line:
                    keep = 1
                else:
                    keep = np.random.binomial(1, 1 - self.knowledge_dropout)
                if keep:
                    to_vectorize.append(line)
        observation.force_set('memory_vecs', [self._vectorize_text(line, truncate=self.knowledge_truncate) for line in to_vectorize])
        return observation

    def load(self, path):
        """
        Return opt and model states.

        Override this method from TorchAgent to allow us to load partial weights from
        pre-trained models.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if 'model' in states:
            new_state_dict = states['model']
            current_state = self.model.state_dict()
            pre_trained_state = {k: v for k, v in new_state_dict.items() if k in current_state}
            current_state.update(pre_trained_state)
            self.model.load_state_dict(current_state)
        if 'optimizer' in states and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(states['optimizer'])
        return states