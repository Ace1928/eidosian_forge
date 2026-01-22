from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
import copy
import json
import os
class SentenceTeacher(IndexTeacher):
    """
    Teacher where the label(s) are the sentences that contain the true answer.

    Some punctuation may be removed from the context and the answer for
    tokenization purposes.

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
        agent = argparser.add_argument_group('SQuAD Sentence Teacher Arguments')
        agent.add_argument('--include-context', type='bool', default=False, help='include context within text instead of as a separate field')

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        context = paragraph['context']
        question = qa['question']
        answers = [a['text'] for a in qa['answers']]
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace('.', '').replace('?', '').replace('!', '')
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)
        edited_sentences = self.sent_tok.tokenize(context)
        labels = []
        label_starts = []
        for sentence in edited_sentences:
            for answer in edited_answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))
                    break
        action = {'context': context, 'text': question, 'labels': labels, 'label_candidates': edited_sentences, 'episode_done': True, 'answer_starts': label_starts}
        if self.include_context:
            action['text'] = action['context'] + '\n' + action['text']
            del action['context']
        return action