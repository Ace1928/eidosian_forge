import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map

    Generates a random link from a model that starts with a random
    4-valent planar graph sampled with the uniform distribution by
    Schaeffer's `PlanarMap program.
    <http://www.lix.polytechnique.fr/~schaeffe/PagesWeb/PlanarMap/index-en.html>`_

    The ``crossings`` argument specifies the number of vertices of the
    initial planar graph G; the number of crossing in the returned knot
    will typically be less. The meanings of the optional arguments are as
    follows:

    1. ``num_components``: The number of components of the returned link.
       The link naively associated to G may have too few or too many
       components. The former situation is resolved by picking another G,
       and the latter by either

       a. Taking the sublink consisting of the components with the largest
          self-crossing numbers.

       b. Resampling G until the desired number of components is achieved;
          this can take a very long time as the expected number of
          components associated to G grows linearly in the number of
          vertices.

       When the argument ``initial_map_gives_link`` is ``False`` the
       program does (a) and this is the default behavior. If you want (b)
       set this argument to ``True``.

       To get the entire link associated to G, set ``num_components`` to
       ```any```, which is also the default.

    2. The 4-valent vertices of G are turned into crossings by flipping a
       fair coin. If you want the unique alternating diagram associated to
       G, pass ``alternating = True``.  If you want there to be no
       obvious Type II Reidemeister moves, pass
       ``consistent_twist_regions = False``.

    3. ``simplify``: Whether and how to try to reduce the number of
       crossings of the link via Reidemeister moves using the method
       ``Link.simplify``.  For no simplification, set ``simplify = None``;
       otherwise set ``simplify`` to be the appropriate mode for
       ``Link.simplify``, for example ``basic`` (the default), ``level``,
       or ``global``.

    4. ``prime_decomposition``:  The initial link generated from G may not
       be prime (and typically isn't if ``initial_map_gives_link`` is
       ``False``). When set (the default), the program undoes any connect
       sums that are "diagrammatic obvious", simplifies the result, and
       repeats until pieces are "diagrammatically prime".  If
       ``return_all_pieces`` is ``False`` (the default) then only the
       largest (apparently) prime component is returned; otherwise all
       summands are returned as a list.

       Warning: If ``prime_decomposition=True`` and
       ``return_all_pieces=False``, then the link returned may have
       fewer components than requested.  This is because a prime piece
       can have fewer components than the link as a whole.


    Some examples:

    >>> K = random_link(25, num_components=1, initial_map_gives_link=True, alternating=True)
    >>> K
    <Link: 1 comp; 25 cross>

    >>> L= random_link(30, consistent_twist_regions=True, simplify = 'global')
    >>> isinstance(random_link(30, return_all_pieces=True), list)
    True
    