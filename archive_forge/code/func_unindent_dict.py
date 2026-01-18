import sys
def unindent_dict(docdict):
    """ Unindent all strings in a docdict """
    can_dict = {}
    for name, dstr in docdict.items():
        can_dict[name] = unindent_string(dstr)
    return can_dict