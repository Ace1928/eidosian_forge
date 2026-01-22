import numpy as np
import pennylane as qml
Transforms one- and two-body terms in physicists' notation to `chemists' notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_\ .

    This converts the input two-body tensor :math:`h_{pqrs}` that constructs :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s`
    to a transformed two-body tensor :math:`V_{pqrs}` that follows the chemists' convention to construct :math:`\sum_{pqrs} V_{pqrs} a^\dagger_p a_q a^\dagger_r a_s`
    in the spatial basis. During the tranformation, some extra one-body terms come out. These are returned as a one-body tensor :math:`T_{pq}` in the
    chemists' notation either as is or after summation with the input one-body tensor :math:`h_{pq}`, if provided.

    Args:
        one_body_tensor (array[float]): a one-electron integral tensor giving the :math:`h_{pq}`.
        two_body_tensor (array[float]): a two-electron integral tensor giving the :math:`h_{pqrs}`.
        spatial_basis (bool): True if the integral tensor are passed in spatial-orbital basis. False if they are in spin basis.

    Returns:
        tuple(array[float], array[float]) or tuple(array[float],): transformed one-body tensor :math:`T_{pq}` and two-body tensor :math:`V_{pqrs}` for the provided terms.

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> qml.qchem.factorization._chemist_transform(two_body_tensor=two, spatial_basis=True)
    (tensor([[-0.427983, -0.      ],
             [-0.      , -0.439431]], requires_grad=True),
    tensor([[[[0.337378, 0.      ],
             [0.       , 0.331856]],
             [[0.      , 0.090605],
             [0.090605 , 0.      ]]],
            [[[0.      , 0.090605],
             [0.090605 , 0.      ]],
             [[0.331856, 0.      ],
             [0.       , 0.348826]]]], requires_grad=True))

    .. details::
        :title: Theory

        The two-electron integral in physicists' notation is defined as:

        .. math::

            \langle pq \vert rs \rangle = h_{pqrs} = \int \frac{\chi^*_{p}(x_1) \chi^*_{q}(x_2) \chi_{r}(x_1) \chi_{s}(x_2)}{|r_1 - r_2|} dx_1 dx_2,

        while in chemists' notation it is written as:

        .. math::

            [pq \vert rs] = V_{pqrs} = \int \frac{\chi^*_{p}(x_1) \chi_{q}(x_1) \chi^*_{r}(x_2) \chi_{s}(x_2)}{|r_1 - r_2|} dx_1 dx_2.

        In the spin basis, this index reordering :math:`pqrs \rightarrow psrq` leads to formation of one-body terms :math:`h_{prrs}` that come out during
        the coversion:

        .. math::

            h_{prrs} = \int \frac{\chi^*_{p}(x_1) \chi^*_{r}(x_2) \chi_{r}(x_1) \chi_{s}(x_2)}{|x_1 - x_2|} dx_1 dx_2,

        where both :math:`\chi_{r}(x_1)` and :math:`\chi_{r}(x_2)` will have same spin functions, i.e.,
        :math:`\chi_{r}(x_i) = \phi(r_i)\alpha(\omega)` or :math:`\chi_{r}(x_i) = \phi(r_i)\beta(\omega)`\ . These are added to the one-electron
        integral tensor :math:`h_{pq}` to compute :math:`T_{pq}`\ .

    