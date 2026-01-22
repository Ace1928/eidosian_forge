from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def _pdict_list(self, other, num):
    """Returns a list of points that gives the shortest path with respect
        to position, velocity, or acceleration from this point to the provided
        point.

        Parameters
        ==========
        other : Point
            A point that may be related to this point by position, velocity, or
            acceleration.
        num : integer
            0 for searching the position tree, 1 for searching the velocity
            tree, and 2 for searching the acceleration tree.

        Returns
        =======
        list of Points
            A sequence of points from self to other.

        Notes
        =====

        It is not clear if num = 1 or num = 2 actually works because the keys
        to ``_vel_dict`` and ``_acc_dict`` are :class:`ReferenceFrame` objects
        which do not have the ``_pdlist`` attribute.

        """
    outlist = [[self]]
    oldlist = [[]]
    while outlist != oldlist:
        oldlist = outlist[:]
        for i, v in enumerate(outlist):
            templist = v[-1]._pdlist[num].keys()
            for i2, v2 in enumerate(templist):
                if not v.__contains__(v2):
                    littletemplist = v + [v2]
                    if not outlist.__contains__(littletemplist):
                        outlist.append(littletemplist)
    for i, v in enumerate(oldlist):
        if v[-1] != other:
            outlist.remove(v)
    outlist.sort(key=len)
    if len(outlist) != 0:
        return outlist[0]
    raise ValueError('No Connecting Path found between ' + other.name + ' and ' + self.name)