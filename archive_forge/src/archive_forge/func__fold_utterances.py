from .build import build
from parlai.core.teachers import DialogTeacher
import json
import os
@staticmethod
def _fold_utterances(raw_dialog):
    dialog = []
    for utterance in raw_dialog:
        if len(dialog) > 0 and dialog[-1]['userId'] == utterance['userId']:
            dialog[-1]['text'] = dialog[-1]['text'] + '\n' + utterance['text']
        else:
            dialog.append({'text': utterance['text'], 'userId': utterance['userId']})
    return dialog