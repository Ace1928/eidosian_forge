import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def DT_drilling_surface(code, make_prefix_unique=True):
    """ Returns a Surface which can be used to construct a triangulation of the
	complement of the knot with Dowker--Thistlethwaite code 'code'. 
	
	Example: DT_drilling_surface([6,10,8,12,4,2]) returns a Surface that can be
	used to construct the 6_1 knot complement. """
    code = list(code)
    signs = code_to_sign_sequence(code)
    contents = ['# A Twister surface file']
    contents.append('#')
    contents.append('# Surface file for knot with Dowker code:')
    contents.append('#    ' + ','.join(map(str, code)))
    contents.append('#')
    contents.append('# To build this knot complement, make a Heegaard splitting with a 2-handle attached above')
    contents.append('# and below every annulus and drill every rectangle exactly once, making sure to drill all')
    contents.append('# of the "x" rectangles before drilling ANY of the "y" rectangles.')
    contents.append('#')
    num_crossings = len(code)
    padded_length = len(str(2 * num_crossings)) if make_prefix_unique else 0
    num_squares = num_crossings * 5
    pairs = list(zip(range(1, 2 * num_crossings + 1, 2), code))
    pairs_dict = dict([(x, abs(y)) for x, y in pairs] + [(abs(y), x) for x, y in pairs])
    signs_dict = dict([(x, y > 0) for x, y in pairs] + [(abs(y), y > 0) for x, y in pairs])
    drill_names = []
    handle_names = []
    for i in range(1, 2 * num_crossings + 1):
        if i % 2 == 1:
            k = i // 2
            cells = ['-' + str(k * 5 + 1), '+' + str(k * 5 + 2), '+' + str(k * 5 + 3)]
        else:
            k = pairs_dict[i] // 2
            if signs[k] == 1:
                cells = ['-' + str(k * 5 + 4), '-' + str(k * 5 + 2), '+' + str(k * 5 + 0)]
            else:
                cells = ['-' + str(k * 5 + 0), '-' + str(k * 5 + 2), '+' + str(k * 5 + 4)]
        over = (i % 2 == 0) ^ signs_dict[i]
        name = ('y' if (i % 2 == 0) ^ signs_dict[i] else 'x') + str(i).zfill(padded_length)
        inverse_name = name.swapcase()
        contents.append(','.join(['rectangle', name, inverse_name] + cells) + '#')
        drill_names.append('!' + name)
    for i in range(1, 2 * num_crossings + 1):
        j = i % (2 * num_crossings) + 1
        if i % 2 == 1:
            k = i // 2
            l = pairs_dict[j] // 2
            if signs[l] == 1:
                cells = ['-' + str(k * 5 + 3), '+' + str(l * 5 + 4)]
            else:
                cells = ['-' + str(k * 5 + 3), '-' + str(l * 5 + 4)]
        else:
            k = pairs_dict[i] // 2
            l = j // 2
            if signs[k] == 1:
                cells = ['-' + str(k * 5 + 0), '+' + str(l * 5 + 1)]
            else:
                cells = ['+' + str(k * 5 + 0), '+' + str(l * 5 + 1)]
        name = 'a' + '_' + str(i).zfill(padded_length)
        inverse_name = name.swapcase()
        contents.append(','.join(['annulus', name, inverse_name] + cells) + '#')
        handle_names.append(name)
        handle_names.append(inverse_name)
    contents.append('# We also give a macro to drill all of the rectangles in an acceptable order.')
    contents.append('# And one for attaching all handles too.')
    drill_names.sort()
    contents.append('macro,s,' + '*'.join(drill_names) + '#')
    contents.append('macro,h,' + '*'.join(handle_names) + '#')
    return Surface('\n'.join(contents))