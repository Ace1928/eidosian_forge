import os
import textwrap
from passlib.utils.compat import irange
def write_expand_function(write, indent=0):
    write(indent, '        def expand(self, key_words):\n            """unrolled version of blowfish key expansion"""\n            ##assert len(key_words) >= 18, "size of key_words must be >= 18"\n\n            P, S = self.P, self.S\n            S0, S1, S2, S3 = S\n\n            #=============================================================\n            # integrate key\n            #=============================================================\n        ')
    for i in irange(18):
        write(indent + 1, '            p%(i)d = P[%(i)d] ^ key_words[%(i)d]\n        ', i=i)
    write(indent + 1, '\n        #=============================================================\n        # update P\n        #=============================================================\n\n        #------------------------------------------------\n        # update P[0] and P[1]\n        #------------------------------------------------\n        l, r = p0, 0\n\n        ')
    render_encipher(write, indent + 1)
    write(indent + 1, '\n        p0, p1 = l, r = r ^ p17, l\n\n        ')
    for i in irange(2, 18, 2):
        write(indent + 1, '            #------------------------------------------------\n            # update P[%(i)d] and P[%(i1)d]\n            #------------------------------------------------\n            l ^= p0\n\n            ', i=i, i1=i + 1)
        render_encipher(write, indent + 1)
        write(indent + 1, '            p%(i)d, p%(i1)d = l, r = r ^ p17, l\n\n            ', i=i, i1=i + 1)
    write(indent + 1, '\n        #------------------------------------------------\n        # save changes to original P array\n        #------------------------------------------------\n        P[:] = (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,\n          p10, p11, p12, p13, p14, p15, p16, p17)\n\n        #=============================================================\n        # update S\n        #=============================================================\n\n        for box in S:\n            j = 0\n            while j < 256:\n                l ^= p0\n\n        ')
    render_encipher(write, indent + 3)
    write(indent + 3, '\n                box[j], box[j+1] = l, r = r ^ p17, l\n                j += 2\n        ')