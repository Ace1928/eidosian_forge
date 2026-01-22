from __future__ import print_function
from pandas import concat, read_csv
from argparse import ArgumentParser, FileType
from numpy import empty
def _lists_to_scipy(onsets_list):
    """
    Inputs:
      * List of dicts (one dict for each condition)

        [{'name':'Low','durations':0,'onsets':[1,3,5]},
         {'name':'Hi', 'durations':0, 'onsets':[2,4,6]}]

        - Or with Parametric Modulators -
        [{'name':'Low','durations':0,'onsets':[1,3,5], 'pmod':[
           {'name': 'RT', 'poly':1, 'param':[42,13,666]}]},
         {'name':'High',, 'durations':0, 'onsets':[2,4,6]}]

    Outputs:
      * Dict of scipy arrays for keys names, durations and onsets
        that can be written using scipy.io.savemat
    """
    conditions_n = len(onsets_list)
    names = empty((conditions_n,), dtype='object')
    durations = empty((conditions_n,), dtype='object')
    onsets = empty((conditions_n,), dtype='object')
    pmoddt = [('name', 'O'), ('poly', 'O'), ('param', 'O')]
    pmods = empty(conditions_n, dtype=pmoddt)
    has_pmods = False
    for i, ons in enumerate(onsets_list):
        names[i] = ons['name']
        durations[i] = ons['durations']
        onsets[i] = ons['onsets']
        if 'pmod' not in ons.keys():
            pmods[i]['name'], pmods[i]['poly'], pmods[i]['param'] = ([], [], [])
        else:
            has_pmods = True
            cond_pmod_list = ons['pmod']
            current_condition_n_pmods = len(cond_pmod_list)
            pmod_names = empty((current_condition_n_pmods,), dtype='object')
            pmod_param = empty((current_condition_n_pmods,), dtype='object')
            pmod_poly = empty((current_condition_n_pmods,), dtype='object')
            for pmod_i, val in enumerate(cond_pmod_list):
                pmod_names[pmod_i] = val['name']
                pmod_param[pmod_i] = val['param']
                pmod_poly[pmod_i] = float(val['poly'])
            pmods[i]['name'] = pmod_names
            pmods[i]['poly'] = pmod_poly
            pmods[i]['param'] = pmod_param
    scipy_onsets = dict(names=names, durations=durations, onsets=onsets)
    if has_pmods:
        scipy_onsets['pmod'] = pmods
    return scipy_onsets