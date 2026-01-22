import datasets
import re
def _process_doc(doc):
    ctx = doc['ctx_a'] + ' ' + doc['ctx_b'].capitalize()
    out_doc = {'query': preprocess(doc['activity_label'] + ': ' + ctx), 'choices': [preprocess(ending) for ending in doc['endings']], 'gold': int(doc['label'])}
    return out_doc