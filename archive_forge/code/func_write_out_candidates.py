import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def write_out_candidates(db, dpath, dtype):
    emotes = {}
    acts = {}
    cands['emote'] = []
    cands['action'] = []
    cands['speech'] = []
    cands['which'] = ['none', 'action', 'emote']
    for d in db:
        gen_no_affordance_actions_for_dialogue(d)
        if d['split'] != dtype:
            continue
        for e in d['emote']:
            if e is not None and e not in emotes:
                cands['emote'].append(e)
                emotes[e] = True
        for act in d['action']:
            if act is not None and act not in acts:
                cands['action'].append(act)
                acts[act] = True
        for s in d['speech']:
            if s is not None:
                cands['speech'].append(s)
    for l in ['emote', 'speech', 'action', 'which']:
        fw = io.open(os.path.join(dpath, l + '_' + dtype + '_cands.txt'), 'w')
        for k in cands[l]:
            fw.write(k + '\n')
        fw.close