from .mcomplex_base import *
from .t3mlite import simplex
def add_choose_generators_info(self, choose_generators_info):
    """
        Expects the result of Manifold._choose_generators_info().

        Adds GeneratorsInfo to each tetrahedron. This encodes generator_index and
        generator_status of a SnapPea Triangulation as described in
        choose_generators.c. However, we only store one number, its absolute value
        giving the generator_index and its sign the generator_status.

        We also set ChooseGenInitialTet of the Mcomplex to be what SnapPea
        considers the base tetrahedron when computing the vertices of the
        fundamental domain.

        We also add the vertices of a fundamental domain as given by the SnapPea
        kernel as SnapPeaIdealVertices to each tetrahedron. We care about these
        numbers when orienting the base tetrahedron to have consistency with the
        SnapPea kernel.
        """
    for tet, info in zip(self.mcomplex.Tetrahedra, choose_generators_info):
        tet.SnapPeaIdealVertices = dict(zip(simplex.ZeroSubsimplices, _clean_ideal_vertices(info['corners'])))
        tet.GeneratorsInfo = dict(zip(simplex.TwoSubsimplices, info['generators']))
        if info['generator_path'] == -1:
            self.mcomplex.ChooseGenInitialTet = tet