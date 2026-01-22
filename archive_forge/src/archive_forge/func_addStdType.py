from pyparsing import *
def addStdType(t, namespace=''):
    fullname = namespace + '_' + t if namespace else t
    typemap[t] = fullname
    user_defined_types.add(t)