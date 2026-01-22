import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def LP_surface(genus, boundary, make_prefix_unique=True):
    """ Returns the contents of a surface file for the surface S_{genus, boundary}.
	We generally follow the naming convention given in Figure 13 
	of the Labruere and Paris paper "Presentations for the punctured 
	mapping class groups in terms of Artin groups".
	
	When genus == 1, the loop a_n is dropped as it is isotopic to the
	loop a_0.
	
	The case of genus == 0, given here, is a small modification of
	Figure 11 in [LP].
	
	You should not use this function directly, rather just call
	Surface((genus, boundary)). """
    contents = ['# A Twister surface file produced by LP_surface.']
    contents.append('# with generating set for MCG(S_{%d,%d}) following Figure 13 of Labruere and' % (genus, boundary))
    contents.append('# Paris paper "Presentations for the punctured mapping class groups')
    contents.append('# in terms of Artin groups".')
    contents.append('#')
    padded_length = len(str(max(2 * genus, boundary))) if make_prefix_unique else 0
    square_count = 0
    if genus == 0:
        if boundary == 0:
            square_count += 1
            contents.append('annulus,a1,A1,-0,+1#')
            contents.append('annulus,a2,A2,-1,+0#')
        else:
            for i in range(boundary):
                square_count += 1
                start = 'rectangle,t' + str(i + 1).zfill(padded_length) + ',T' + str(i + 1).zfill(padded_length)
                connections = ',-' + str(square_count - 1) + ',+' + str(square_count)
                contents.append(start + connections + '#')
                if i == boundary - 1:
                    start = 'annulus,a' + str(i + 1).zfill(padded_length) + ',A' + str(i + 1).zfill(padded_length)
                    connections = ',-' + str(square_count) + ',+' + '0'
                    contents.append(start + connections + '#')
                else:
                    square_count += 1
                    start = 'annulus,a' + str(i + 1).zfill(padded_length) + ',A' + str(i + 1).zfill(padded_length)
                    connections = ',-' + str(square_count - 1) + ',+' + str(square_count)
                    contents.append(start + connections + '#')
    else:
        c_loop = -1
        next_line = 'annulus,a' + '0'.zfill(padded_length) + ',A' + '0'.zfill(padded_length) + ',+0'
        contents.append(next_line + '#')
        start = 'annulus,b' + '1'.zfill(padded_length) + ',B' + '1'.zfill(padded_length)
        connections = ',-0'
        if boundary > 1:
            for i in range(boundary - 1):
                square_count += 1
                connections = connections + ',-' + str(square_count)
                square_count += 1
                start2 = 'annulus,a' + str(i + 1).zfill(padded_length) + ',A' + str(i + 1).zfill(padded_length)
                connections2 = ',+' + str(square_count - 1) + ',+' + str(square_count)
                contents.append(start2 + connections2 + '#')
                start3 = 'rectangle,t' + str(i + 1).zfill(padded_length) + ',T' + str(i + 1).zfill(padded_length)
                connections3 = ',-' + str(square_count)
                contents.append(start3 + connections3 + '#')
        elif boundary == 1:
            square_count += 1
            connections = connections + ',-' + str(square_count)
            start2 = 'rectangle,t' + '1'.zfill(padded_length) + ',T' + '1'.zfill(padded_length)
            connections2 = ',+' + str(square_count)
            contents.append(start2 + connections2 + '#')
        elif boundary == 0 and genus > 1:
            square_count += 1
            connections = connections + ',+' + str(square_count)
        if genus > 1 and boundary > 0:
            square_count += 1
            connections = connections + ',-' + str(square_count)
            start2 = 'annulus,a' + str(boundary).zfill(padded_length) + ',A' + str(boundary).zfill(padded_length)
            connections2 = ',+' + str(square_count)
            contents.append(start2 + connections2 + '#')
            square_count += 1
            connections = connections + ',+' + str(square_count)
        contents.append(start + connections + '#')
        for i in range(2, 2 * genus - 1):
            square_count += 1
            start = 'annulus,b' + str(i).zfill(padded_length) + ',B' + str(i).zfill(padded_length)
            connections = ',-' + str(square_count - 1) + ',+' + str(square_count)
            contents.append(start + connections + '#')
            if i == 2:
                c_loop = len(contents)
        if genus > 1:
            square_count += 1
            start = 'annulus,b' + str(2 * genus - 1).zfill(padded_length) + ',B' + str(2 * genus - 1).zfill(padded_length)
            connections = ',-' + str(square_count - 1)
            contents.append(start + connections + '#')
        if c_loop > -1:
            next_line = 'annulus,c,C,-' + str(square_count)
            contents.append(next_line + '#')
            contents[c_loop] = contents[c_loop][:-1] + ',+' + str(square_count) + '#'
    return '\n'.join(contents)