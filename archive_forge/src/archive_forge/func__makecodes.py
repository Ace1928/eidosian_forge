from _sre import MAXREPEAT, MAXGROUPS
def _makecodes(*names):
    items = [_NamedIntConstant(i, name) for i, name in enumerate(names)]
    globals().update({item.name: item for item in items})
    return items