from fontTools.cffLib import maxStackLimit
def _flattenBlendArgs(args):
    token_list = []
    for arg in args:
        if isinstance(arg, list):
            token_list.extend(arg)
            token_list.append('blend')
        else:
            token_list.append(arg)
    return token_list