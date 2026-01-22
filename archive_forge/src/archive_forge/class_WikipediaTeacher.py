from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from .build import build
import copy
import json
import os
class WikipediaTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        if not hasattr(self, 'prefix'):
            self.prefix = ''
            self.suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        if not hasattr(self, 'no_evidence'):
            self.no_evidence = False
        qa_dir, self.evidence_dir = _path(opt)
        opt['datafile'] = os.path.join(qa_dir, self.prefix + 'wikipedia-' + self.suffix + '.json')
        self.id = 'triviaqa'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            data = json.load(data_file)['Data']
        for datapoint in data:
            question = datapoint['Question']
            answers = [datapoint['Answer']['Value']] + sorted(list(set(datapoint['Answer']['Aliases'])))
            evidence_list = datapoint['EntityPages']
            if self.no_evidence:
                yield ((question, answers), True)
            else:
                if len(evidence_list) == 0:
                    continue
                evidence = ''
                for evidence_item in evidence_list:
                    evidence_file_path = os.path.join(self.evidence_dir, 'wikipedia', evidence_item['Filename'])
                    with open(evidence_file_path) as evidence_file:
                        evidence += 'Title: %s\n' % evidence_item['Title']
                        evidence += evidence_file.read() + '\n\n'
                yield ((evidence + question, answers), True)