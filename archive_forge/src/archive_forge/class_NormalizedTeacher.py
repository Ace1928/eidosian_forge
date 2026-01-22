from parlai.core.teachers import FbDialogTeacher
from parlai.utils.misc import warn_once
from .build import build
from parlai.utils.strings import normalize_reply
import parlai.utils.logging as logging
import copy
import os
class NormalizedTeacher(SelfOriginalTeacher):

    def normalize_replies(self, x):
        xs = x.split('\n')
        xs2 = []
        for x in xs:
            if 'your persona:' in x:
                x = x[len('your persona: '):]
                x = normalize_reply(x)
                x = 'your persona: ' + x
            else:
                x = normalize_reply(x)
            xs2.append(x)
        return '\n'.join(xs2)

    def setup_data(self, path):
        logging.info(f'loading normalized fbdialog data: {path}')
        for (text, labels, reward, candidates), new_episode in super().setup_data(path):
            text = self.normalize_replies(text)
            labels = [self.normalize_replies(l) for l in labels]
            candidates = [self.normalize_replies(l) for l in labels]
            yield ((text, labels, reward, candidates), new_episode)