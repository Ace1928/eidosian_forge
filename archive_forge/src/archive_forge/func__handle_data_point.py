from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os
import json
def _handle_data_point(data_point):
    output = []
    context_question_txt = ''
    for [title, sentences_list] in data_point['context']:
        sentences = '\\n'.join(sentences_list)
        context_question_txt += '{}\\n{}\\n\\n'.format(title, sentences)
    context_question_txt += data_point['question']
    output = OUTPUT_FORMAT.format(context_question=context_question_txt, answer=data_point['answer'])
    output += '\t\tepisode_done:True\n'
    return output