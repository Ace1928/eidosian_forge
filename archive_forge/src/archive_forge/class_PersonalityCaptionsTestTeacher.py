from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
import json
import os
class PersonalityCaptionsTestTeacher(PersonalityCaptionsTeacher):
    """
    Test PersonalityCaptions teacher for ensuring pretrained model does not break.
    """

    def _setup_data(self, data_path, personalities_data_path):
        super()._setup_data(data_path, personalities_data_path)
        from parlai.zoo.personality_captions.transresnet import download
        download(self.opt['datapath'])
        image_features_path = os.path.join(self.opt['datapath'], 'models/personality_captions/transresnet/test_image_feats')
        import torch
        self.image_features = torch.load(image_features_path)

    def reset(self):
        """
        Reset teacher.
        """
        super().reset()
        self.example = None

    def num_episodes(self):
        """
        Return number of episodes.
        """
        return len(self.image_features)

    def num_examples(self):
        """
        Return number of examples.
        """
        return len(self.image_features)

    def get(self, episode_idx, entry_idx=0):
        """
        Get an example.

        :param episode_idx:
            index of episode in self.data
        :param entry_idx:
            optional, which entry in the episode to get

        :return:
            an example
        """
        data = self.data[episode_idx]
        action = {'text': data['personality'] if self.include_personality else '', 'image': self.image_features[data['image_hash']], 'episode_done': True, 'labels': [data['comment']]}
        if self.num_test_labels == 5 and 'test' in self.datatype:
            action['labels'] += data['additional_comments']
        if 'candidates' in data:
            if self.num_test_labels == 5 and 'test' in self.datatype:
                action['label_candidates'] = data['500_candidates']
            else:
                action['label_candidates'] = data['candidates']
        return action