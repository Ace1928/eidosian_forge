from .build import build
from parlai.core.teachers import DialogTeacher
import json
import os
@staticmethod
def _create_learning_examples(opponent_utterances, answer_utterances):
    examples = [u for u in map(lambda pair: ((pair[0]['text'], [pair[1]['text']]), False), zip(opponent_utterances, answer_utterances))]
    return examples