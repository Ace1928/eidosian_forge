import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def write_alldata(opt, db, dpath, ltype, split):
    fname = os.path.join(dpath, ltype + '_' + split + '.txt')
    fw_tst = io.open(fname, 'w')
    for d in db:
        if d['split'] != split:
            continue
        d = d.copy()
        d['self_agent'] = d['agents'][1]
        d['partner_agent'] = d['agents'][0]
        write_dialog(opt, fw_tst, d, ltype, split)
        d2 = d.copy()
        d2['self_agent'] = d2['agents'][0]
        d2['partner_agent'] = d2['agents'][1]
        d2['speech'] = list(d2['speech'])
        d2['speech'].insert(0, None)
        d2['emote'] = list(d2['emote'])
        d2['emote'].insert(0, None)
        d2['action'] = list(d2['action'])
        d2['action'].insert(0, None)
        d2['available_actions'] = list(d2['available_actions'])
        d2['available_actions'].insert(0, None)
        d2['no_affordance_actions'] = list(d2['no_affordance_actions'])
        d2['no_affordance_actions'].insert(0, None)
        write_dialog(opt, fw_tst, d2, ltype, split)
    fw_tst.close()