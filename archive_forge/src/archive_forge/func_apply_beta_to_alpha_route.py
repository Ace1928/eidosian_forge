from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def apply_beta_to_alpha_route(alpha_implications, beta_rules):
    """apply additional beta-rules (And conditions) to already-built
    alpha implication tables

       TODO: write about

       - static extension of alpha-chains
       - attaching refs to beta-nodes to alpha chains


       e.g.

       alpha_implications:

       a  ->  [b, !c, d]
       b  ->  [d]
       ...


       beta_rules:

       &(b,d) -> e


       then we'll extend a's rule to the following

       a  ->  [b, !c, d, e]
    """
    x_impl = {}
    for x in alpha_implications.keys():
        x_impl[x] = (set(alpha_implications[x]), [])
    for bcond, bimpl in beta_rules:
        for bk in bcond.args:
            if bk in x_impl:
                continue
            x_impl[bk] = (set(), [])
    seen_static_extension = True
    while seen_static_extension:
        seen_static_extension = False
        for bcond, bimpl in beta_rules:
            if not isinstance(bcond, And):
                raise TypeError('Cond is not And')
            bargs = set(bcond.args)
            for x, (ximpls, bb) in x_impl.items():
                x_all = ximpls | {x}
                if bimpl not in x_all and bargs.issubset(x_all):
                    ximpls.add(bimpl)
                    bimpl_impl = x_impl.get(bimpl)
                    if bimpl_impl is not None:
                        ximpls |= bimpl_impl[0]
                    seen_static_extension = True
    for bidx, (bcond, bimpl) in enumerate(beta_rules):
        bargs = set(bcond.args)
        for x, (ximpls, bb) in x_impl.items():
            x_all = ximpls | {x}
            if bimpl in x_all:
                continue
            if any((Not(xi) in bargs or Not(xi) == bimpl for xi in x_all)):
                continue
            if bargs & x_all:
                bb.append(bidx)
    return x_impl