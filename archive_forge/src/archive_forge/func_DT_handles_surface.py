import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def DT_handles_surface(code, make_prefix_unique=True):
    """ Returns a Surface which can be used to construct a triangulation of the
	complement of the knot with Dowker code 'code'. 
	
	Example: DT_handles_surface([6,10,8,12,4,2]) returns a Surface that can be
	used to construct the 6_1 knot complement. """
    code = list(code)
    signs = code_to_sign_sequence(code)
    contents = ['# A Twister surface file']
    contents.append('#')
    contents.append('# Surface file for knot with Dowker code:')
    contents.append('#    ' + ','.join(map(str, code)))
    contents.append('#')
    contents.append('# To build this knot complement, make a Heegaard splitting with a 2-handle attached above')
    contents.append('# and below the sequence of annuli specified in the macro "h".')
    contents.append('#')
    num_crossings = len(code)
    pairs = list(zip(range(1, 2 * num_crossings + 1, 2), code))
    pairs_dict = dict([(x, abs(y)) for x, y in pairs] + [(abs(y), x) for x, y in pairs])
    signs_dict = dict([(x, y > 0) for x, y in pairs] + [(abs(y), y > 0) for x, y in pairs])
    overcrossing = [signs_dict[i + 1] ^ (i % 2 == 1) for i in range(2 * num_crossings)]
    where_switch = list(filter(lambda i: not signs_dict[i + 1] ^ signs_dict[(i + 1) % (2 * num_crossings) + 1], range(2 * num_crossings)))
    last_true = max(where_switch) + 1
    num_annuli = len(where_switch)
    squares_in_crossings = 4 * num_crossings
    squares_in_links = 2 * num_annuli
    squares_in_rectangles = 2 * num_annuli
    num_squares = squares_in_crossings + squares_in_links + squares_in_rectangles
    padded_length = len(str(num_annuli)) if make_prefix_unique else 0
    annuli_count = 0
    squares = []
    back_squares = []
    handle_names = []
    for j in range(2 * num_crossings):
        i = (j + last_true) % (2 * num_crossings)
        if i % 2 == 0:
            k = i // 2
            squares.append('-' + str(4 * k + 0))
            squares.append('+' + str(4 * k + 1))
            back_squares.append('+' + str(4 * k + 3))
            back_squares.append('-' + str(4 * k + 2))
        else:
            k = pairs_dict[i + 1] // 2
            if signs[k] == +1:
                squares.append('-' + str(4 * k + 3))
                squares.append('+' + str(4 * k + 0))
                back_squares.append('+' + str(4 * k + 2))
                back_squares.append('-' + str(4 * k + 1))
            else:
                squares.append('-' + str(4 * k + 1))
                squares.append('+' + str(4 * k + 2))
                back_squares.append('+' + str(4 * k + 0))
                back_squares.append('-' + str(4 * k + 3))
        if i in where_switch:
            row = 'annulus,a' + str(annuli_count).zfill(padded_length) + ',A' + str(annuli_count).zfill(padded_length) + ','
            row += '+' + str(squares_in_crossings + 2 * annuli_count) + ','
            row += ','.join(squares) + ','
            row += '-' + str(squares_in_crossings + 2 * ((annuli_count + 1) % num_annuli)) + ','
            row += '-' + str(squares_in_crossings + squares_in_links + 2 * annuli_count) + ','
            row += '+' + str(squares_in_crossings + squares_in_links + 2 * annuli_count + 1) + ','
            row += '+' + str(squares_in_crossings + 2 * ((annuli_count + 1) % num_annuli) + 1) + ','
            row += ','.join(back_squares[::-1]) + ','
            row += '-' + str(squares_in_crossings + 2 * annuli_count + 1)
            row += '#'
            contents.append(row)
            handle_names.append(('a' if overcrossing[i] else 'A') + str(annuli_count).zfill(padded_length))
            annuli_count += 1
            squares = []
            back_squares = []
    for i in range(num_annuli):
        row = 'rectangle,t' + str(i).zfill(padded_length) + ',T' + str(i).zfill(padded_length) + ','
        row += '+' + str(squares_in_crossings + squares_in_links + 2 * i) + ','
        row += '-' + str(squares_in_crossings + squares_in_links + 2 * i + 1)
        row += '#'
        contents.append(row)
    num_squares = num_crossings * 5
    contents.append('# We also give a macro for attaching the required handles.')
    contents.append('macro,h,' + '*'.join(handle_names) + '#')
    return Surface('\n'.join(contents))