import parlai.core.build_data as build_data
import os
import json
from parlai.core.build_data import DownloadableFile
def _parse_answers(q_a):
    starts = []
    labels = []
    for each in q_a['answers']:
        starts.append(str(each['answer_start']))
        labels.append(each['text'].replace('|', ' __PIPE__ '))
    return ('|'.join(starts), '|'.join(labels))