from collections import OrderedDict

     Computes the lexical distance between strings A and B.
     The "distance" between two strings is given by counting the minimum number
     of edits needed to transform string A into string B. An edit can be an
     insertion, deletion, or substitution of a single character, or a swap of two
     adjacent characters.
     This distance can be useful for detecting typos in input or sorting
     @returns distance in number of edits
    