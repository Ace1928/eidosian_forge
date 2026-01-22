from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from .build import build
import json
import os
class TitleTeacher(DefaultTeacher):
    """
    This version of SquAD inherits from the Default Teacher.

    The only
    difference is that the 'text' field of an observation will contain
    the title of the article separated by a newline from the paragraph and the
    query.
    Note: The title will contain underscores, as it is the part of the link for
    the Wikipedia page; i.e., the article is at the site:
    https://en.wikipedia.org/wiki/{TITLE}
    Depending on your task, you may wish to remove underscores.
    """

    def __init__(self, opt, shared=None):
        self.id = 'squad_title'
        build(opt)
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        for article in self.squad:
            title = article['title']
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    question = qa['question']
                    ans_iter = [{'text': ''}]
                    if not qa['is_impossible']:
                        ans_iter = qa['answers']
                    answers = (a['text'] for a in ans_iter)
                    context = paragraph['context']
                    yield (('\n'.join([title, context, question]), answers), True)