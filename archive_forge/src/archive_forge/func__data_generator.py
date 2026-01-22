from .build import build
from parlai.core.teachers import DialogTeacher
import json
import os
@staticmethod
def _data_generator(dialogs_dict):
    for dialog in dialogs_dict:
        folded_dialog = DefaultTeacher._fold_utterances(dialog['thread'])
        context = dialog['context']
        if len(folded_dialog) < 2:
            continue
        u1_utterances = folded_dialog[::2]
        u2_utterances = folded_dialog[1::2]
        it = [((context, ['']), True)] + DefaultTeacher._create_learning_examples(u1_utterances, u2_utterances)
        for second_user_examples in it:
            yield second_user_examples
        if len(u1_utterances) > 1:
            examples = [((context, [u1_utterances[0]['text']]), True)] + DefaultTeacher._create_learning_examples(u2_utterances, u1_utterances[1:])
        else:
            examples = [((context, [u1_utterances[0]['text']]), True)]
        for first_user_examples in examples:
            yield first_user_examples