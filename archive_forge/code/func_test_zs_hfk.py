from .links_base import Strand, Crossing, Link
import random
import collections
def test_zs_hfk(crossings, how_many):
    from networkx.algorithms import approximation
    for _ in range(how_many):
        K = spherogram.random_link(crossings, num_components=1, initial_map_gives_link=True, consistent_twist_regions=True)
        E = K.exterior()
        if not E.solution_type().startswith('all tet'):
            continue
        exhaust = good_exhaustion(K, 100)
        encoding0 = MorseEncoding(exhaust)
        encoding1 = morse_encoding_from_zs_hfk(K)
        M0 = encoding0.link().exterior()
        M1 = encoding1.link().exterior()
        assert orient_pres_isometric(E, M0)
        print(encoding0.width, encoding1.width, orient_pres_isometric(E, M1))