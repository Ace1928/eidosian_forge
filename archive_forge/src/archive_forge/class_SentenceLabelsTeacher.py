from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from .build import build
import json
import os
class SentenceLabelsTeacher(IndexTeacher):
    """
    Teacher which contains the question as the text, the sentences as the label
    candidates, and the label as the sentence containing the answer.

    Some punctuation may be removed for tokenization purposes.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download('punkt')
            self.sent_tok = nltk.data.load(st_path)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        context = paragraph['context']
        question = qa['question']
        answers = ['']
        if not qa['is_impossible']:
            answers = [a['text'] for a in qa['answers']]
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace('.', '').replace('?', '').replace('!', '')
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)
        edited_sentences = self.sent_tok.tokenize(context)
        labels = []
        for sentence in edited_sentences:
            for answer in edited_answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    break
        plausible = []
        if qa['is_impossible']:
            plausible = qa['plausible_answers']
        action = {'id': 'SquadSentenceLabels', 'text': question, 'labels': labels, 'plausible_answers': plausible, 'label_candidates': edited_sentences, 'episode_done': True}
        return action