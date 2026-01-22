import numpy as np
from ase.data import atomic_numbers
from ase.ga.offspring_creator import OffspringCreator
class MoveRightMutation(ElementMutation):
    """
    Mutation that exchanges an element with an element one step
    (or more steps if fewer is forbidden) to the right in the
    same row in the periodic table.

    This mutation is introduced and used in:
    P. B. Jensen et al., Phys. Chem. Chem. Phys., 16, 36, 19732-19740 (2014)

    See MoveDownMutation for the idea behind

    Parameters:

    element_pool: List of elements in the phase space. The elements can be
        grouped if the individual consist of different types of elements.
        The list should then be a list of lists e.g. [[list1], [list2]]

    max_diff_elements: The maximum number of different elements in the
        individual. Default is infinite. If the elements are grouped
        max_diff_elements should be supplied as a list with each input
        corresponding to the elements specified in the same input in
        element_pool.

    min_percentage_elements: The minimum percentage of any element in the
        individual. Default is any number is allowed. If the elements are
        grouped min_percentage_elements should be supplied as a list with
        each input corresponding to the elements specified in the same input
        in element_pool.

    rng: Random number generator
        By default numpy.random.

    Example: element_pool=[[A,B,C,D],[x,y,z]], max_diff_elements=[3,2],
        min_percentage_elements=[.25, .5]
        An individual could be "D,B,B,C,x,x,x,x,z,z,z,z"
    """

    def __init__(self, element_pool, max_diff_elements=None, min_percentage_elements=None, verbose=False, num_muts=1, rng=np.random):
        ElementMutation.__init__(self, element_pool, max_diff_elements, min_percentage_elements, verbose, num_muts=num_muts, rng=rng)
        self.descriptor = 'MoveRightMutation'

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]
        ltbm, choices = self.get_mutation_index_list_and_choices(f)
        ptrow, ptcol = get_row_column(f[ltbm[0]].symbol)
        popped = []
        m = 0
        for j in range(len(choices)):
            e = choices[j - m]
            row, column = get_row_column(e)
            if row != ptrow or column <= ptcol:
                popped.append(choices.pop(j - m))
                m += 1
        used_descriptor = self.descriptor
        if len(choices) == 0:
            msg = '{0},{2} cannot be mutated by {1}, '
            msg = msg.format(f.info['confid'], self.descriptor, f[ltbm[0]].symbol)
            msg += 'doing random mutation instead'
            if self.verbose:
                print(msg)
            used_descriptor = 'RandomElementMutation_from_{0}'
            used_descriptor = used_descriptor.format(self.descriptor)
            self.rng.shuffle(popped)
            choices = popped
        else:
            choices.sort(key=lambda x: get_row_column(x)[1])
        new_element = choices[0]
        for a in f:
            if a.index in ltbm:
                a.symbol = new_element
            indi.append(a)
        return (self.finalize_individual(indi), used_descriptor + ': Parent {0}'.format(f.info['confid']))