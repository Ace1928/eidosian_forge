from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
import copy
import json
import os
class FulldocsentenceTeacher(FulldocTeacher):
    """
    Teacher which contains the question as the text, the sentences as the label
    candidates, and the label as the sentence containing the answer.

    Some punctuation may be removed for tokenization purposes.

    If `include_context` is False, the teacher returns action dict in the
    following format:
    {
        'context': <context>,
        'text': <question>,
        'labels': <sentences containing the true answer>,
        'label_candidates': <all sentences in the context>,
        'episode_done': True,
        'answer_starts': <index of start of answer in context>
    }
    Otherwise, the 'text' field contains <context>
<question> and there is
    no separate context field.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.sent_tok = get_sentence_tokenizer()
        self.include_context = opt.get('include_context', False)

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('SQuAD Fulldoc Sentence Teacher Arguments')
        agent.add_argument('--include-context', type='bool', default=False, help='include context within text instead of as a separate field')

    def get(self, episode_idx, entry_idx=None):
        action = {}
        episode = self.episodes[episode_idx][entry_idx]
        context = ' '.join(episode['text'].split('\n')[:-1]).replace('\xa0', ' ')
        question = episode['text'].split('\n')[-1]
        label_field = 'labels' if 'labels' in episode else 'eval_labels'
        answers = []
        for answer in episode[label_field]:
            new_answer = answer.replace('.', '').replace('?', '').replace('!', '')
            context = context.replace(answer, new_answer)
            answers.append(new_answer)
        sentences = self.sent_tok.tokenize(context)
        labels = []
        label_starts = []
        for sentence in sentences:
            for answer in answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))
        action = {'context': context, 'text': question, label_field: labels, 'answer_starts': label_starts, 'label_candidates': sentences, 'episode_done': episode['episode_done']}
        if self.include_context:
            action['text'] = action['context'] + '\n' + action['text']
            del action['context']
        return action