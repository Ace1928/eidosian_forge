import json
import os
from typing import Tuple
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.utils.typing import TShared
from .build import build
class GenerationTeacher(ImageChatTeacher):
    """
    GenerationTeacher - dialogues are split into two episodes, one from
    each individual person's point of view.

    Used in the #dodecaDialogue task. (see https://parl.ai/projects/dodecadialogue/)
    """

    def __init__(self, opt: Opt, shared: TShared=None):
        if not shared:
            self.idx_to_ep = {}
        else:
            self.idx_to_ep = shared['idx_to_ep']
        self.prepend_personality = opt.get('prepend_personality', True)
        self.include_dialogue_history = opt.get('include_dialogue_history', True)
        super().__init__(opt, shared)
        self.num_eps = len(self.data) + len([d for d in self.data if len(d['dialog']) > 1])

    @staticmethod
    def add_cmdline_args(argparser):
        ImageChatTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('generation teacher arguments')
        agent.add_argument('--prepend-personality', type='bool', default=True, help='if true, always prepend first turn text with the personality')
        agent.add_argument('--include-dialogue-history', type='bool', default=True, help='if false, remove the dialogue history')

    def num_episodes(self) -> int:
        return self.num_eps

    def get(self, episode_idx: int, entry_idx: int=0):
        entry_idx *= 2
        first_turn = entry_idx == 0
        if episode_idx >= len(self.data):
            data = self.data[self.idx_to_ep[episode_idx]]
            entry_idx += 1
        else:
            data = self.data[episode_idx]
        personality, label = data['dialog'][entry_idx]
        if not self.include_personality:
            personality = ''
        if entry_idx > 0:
            _, text = data['dialog'][entry_idx - 1]
            if not self.include_dialogue_history:
                text = ''
            if first_turn and self.prepend_personality and self.include_personality:
                text = '\n'.join([personality, text])
        elif self.prepend_personality and self.include_personality:
            text = personality
        else:
            text = ''
        episode_done = entry_idx >= len(data['dialog']) - 2
        action = {'text': text, 'personality': personality, 'image_id': data['image_hash'], 'episode_done': episode_done, 'labels': [label]}
        if 'candidates' in data:
            action['label_candidates'] = data['candidates'][entry_idx][self.num_cands]
        return action

    def _setup_data(self, data_path: str, personalities_data_path: str):
        super()._setup_data(data_path, personalities_data_path)
        ep_idx = len(self.data)
        for i, d in enumerate(self.data):
            if len(d['dialog']) > 1:
                self.idx_to_ep[ep_idx] = i
                ep_idx += 1

    def share(self):
        shared = super().share()
        shared['idx_to_ep'] = self.idx_to_ep
        return shared