import string
from xml.dom import Node
def _sorter_ns(n1, n2):
    '''_sorter_ns((n,v),(n,v)) -> int
    "(an empty namespace URI is lexicographically least)."'''
    if n1[0] == 'xmlns':
        return -1
    if n2[0] == 'xmlns':
        return 1
    return cmp(n1[0], n2[0])