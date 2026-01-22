from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
import random
class DSTC7Teacher(FixedDialogTeacher):
    """
    Teacher that corresponds to the default DSTC7 ubuntu track 1.

    The data hasn't been augmented by using the multi-turn utterances.
    """

    def __init__(self, opt, shared=None):
        self.split = 'train'
        if 'valid' in opt['datatype']:
            self.split = 'dev'
        if 'test' in opt['datatype']:
            self.split = 'test'
        build(opt)
        basedir = os.path.join(opt['datapath'], 'dstc7')
        filepath = os.path.join(basedir, 'ubuntu_%s_subtask_1%s.json' % (self.split, self.get_suffix()))
        if shared is not None:
            self.data = shared['data']
        else:
            with open(filepath, 'r') as f:
                self.data = json.loads(f.read())
            if self.split == 'test':
                id_to_res = {}
                with open(os.path.join(basedir, 'ubuntu_responses_subtask_1.tsv'), 'r') as f:
                    for line in f:
                        splited = line[0:-1].split('\t')
                        id_ = splited[0]
                        id_res = splited[1]
                        res = splited[2]
                        id_to_res[id_] = [{'candidate-id': id_res, 'utterance': res}]
                for sample in self.data:
                    sample['options-for-correct-answers'] = id_to_res[str(sample['example-id'])]
        super().__init__(opt, shared)
        self.reset()

    def get_suffix(self):
        return ''

    def _setup_data(self, datatype):
        pass

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def get(self, episode_idx, entry_idx=0):
        rand = random.Random(episode_idx)
        episode = self.data[episode_idx]
        texts = []
        for m in episode['messages-so-far']:
            texts.append(m['speaker'].replace('_', ' ') + ': ')
            texts.append(m['utterance'] + '\n')
        text = ''.join(texts)
        labels = [m['utterance'] for m in episode['options-for-correct-answers']]
        candidates = [m['utterance'] for m in episode['options-for-next']]
        if labels[0] not in candidates:
            candidates = labels + candidates
        rand.shuffle(candidates)
        label_key = 'labels' if self.split == 'train' else 'eval_labels'
        return {'text': text, label_key: labels, 'label_candidates': candidates, 'episode_done': True}

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared