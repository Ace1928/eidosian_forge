
    Represents an obstruction cocycle of a pSL(n,C) representation as
    described in Definition 1.7 of

    Garoufalidis, Thurston, Zickert
    The Complex Volume of SL(n,C)-Representations of 3-Manifolds
    https://arxiv.org/abs/1111.2828

    To generate such an obstruction class, call:

    >>> from snappy import Manifold
    >>> M = Manifold("4_1")
    >>> cs = M.ptolemy_obstruction_classes()

    Print out the values:

    >>> for c in cs: print(c)
    PtolemyObstructionClass(s_0_0 - 1, s_1_0 - 1, s_2_0 - 1, s_3_0 - 1, s_0_0 - s_0_1, s_1_0 - s_3_1, s_2_0 - s_2_1, s_3_0 - s_1_1)
    PtolemyObstructionClass(s_0_0 + 1, s_1_0 - 1, s_2_0 - 1, s_3_0 + 1, s_0_0 - s_0_1, s_1_0 - s_3_1, s_2_0 - s_2_1, s_3_0 - s_1_1)
    